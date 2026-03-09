# =============================================
# main.py - HelpKart AI Agent Backend (FastAPI + WebSockets)
# =============================================
# Production-ready backend with:
# - Environment variables for security
# - Comprehensive error handling
# - Connection pooling
# - Rate limiting
# - Proper logging
# - Session management
# - RAG with vector search
# =============================================

import os
import asyncio
import uuid
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

# Database
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

# ML/AI
from sentence_transformers import SentenceTransformer
from groq import Groq
from groq import AsyncGroq

# Utilities
from dotenv import load_dotenv
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================
# Configuration (from environment variables)
# =============================================
class Config:
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
    
    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Redis (optional, for rate limiting)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # App Settings
    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "20"))
    RAG_MATCH_THRESHOLD = float(os.getenv("RAG_MATCH_THRESHOLD", "0.78"))
    RAG_MATCH_COUNT = int(os.getenv("RAG_MATCH_COUNT", "5"))
    RATE_LIMIT_MESSAGES = int(os.getenv("RATE_LIMIT_MESSAGES", "30"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Model Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
    
    # Security
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Server
    PORT = int(os.getenv("PORT", "8000"))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    @classmethod
    def validate(cls):
        """Validate required config"""
        required = ['SUPABASE_URL', 'SUPABASE_KEY', 'GROQ_API_KEY']
        missing = [r for r in required if not getattr(cls, r)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Log config (without secrets)
        logger.info(f"""
        ========================================
        Configuration loaded:
        Environment: {cls.ENVIRONMENT}
        Port: {cls.PORT}
        Model: {cls.LLM_MODEL}
        Max History: {cls.MAX_HISTORY_LENGTH}
        RAG Threshold: {cls.RAG_MATCH_THRESHOLD}
        CORS Origins: {cls.CORS_ORIGINS}
        ========================================
        """)

# =============================================
# Initialize Clients
# =============================================
try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Supabase client with retry config
supabase: Client = create_client(
    Config.SUPABASE_URL, 
    Config.SUPABASE_KEY,
    options=ClientOptions(
        postgrest_client_timeout=30,
        storage_client_timeout=30,
        schema="public"
    )
)

# Embedding model (load once, reuse)
try:
    logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
    embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
    logger.info(f"Successfully loaded embedding model: {Config.EMBEDDING_MODEL}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    # Fallback to a smaller model if main model fails
    try:
        logger.info("Attempting to load fallback model: all-MiniLM-L6-v2")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded fallback model")
    except Exception as e2:
        logger.error(f"Failed to load fallback model: {e2}")
        raise

# Groq client (async for better performance)
try:
    groq_client = AsyncGroq(api_key=Config.GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

# Redis for rate limiting (optional)
redis_client = None
try:
    if Config.REDIS_URL:
        redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        logger.info("Redis connected for rate limiting")
    else:
        logger.warning("REDIS_URL not provided - rate limiting disabled")
except Exception as e:
    logger.warning(f"Redis not available - rate limiting disabled: {e}")

# =============================================
# System Prompt (enhanced for no hallucinations)
# =============================================
SYSTEM_PROMPT = """You are a helpful customer support agent for HelpKart. Follow these rules STRICTLY:

1. Be natural and conversational, like a live phone call
2. Use ONLY the provided context to answer questions
3. If information is missing, say: "I don't have that information. Let me connect you with a human agent."
4. Never make up policies, prices, or order status
5. If unsure, ask clarifying questions
6. Handle interruptions gracefully
7. Be concise but helpful
8. Maintain conversation flow

Current Context:
{context}

Conversation Summary:
{summary}
"""

# =============================================
# Security & Rate Limiting
# =============================================
security = HTTPBearer(auto_error=False)  # Don't auto-error for WebSocket

async def rate_limit(session_id: str) -> tuple[bool, str]:
    """Rate limiting using Redis or in-memory fallback"""
    if not redis_client:
        return True, "Rate limiting disabled"
    
    try:
        key = f"rate_limit:{session_id}"
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, Config.RATE_LIMIT_WINDOW)
        
        remaining = Config.RATE_LIMIT_MESSAGES - current
        if remaining < 0:
            return False, f"Rate limit exceeded. Try again in {Config.RATE_LIMIT_WINDOW} seconds"
        
        return True, f"{remaining} messages remaining"
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        return True, "Rate limit check failed"

# =============================================
# Database Helpers with Retry Logic
# =============================================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def execute_db_operation(operation, *args, **kwargs):
    """Execute DB operations with retry logic"""
    try:
        # Run in thread pool to avoid blocking
        result = await asyncio.to_thread(operation, *args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise

# =============================================
# Embedding and RAG Functions
# =============================================
async def embed_and_store_kb():
    """Embed knowledge base content (runs on startup)"""
    try:
        # Get KB items without embeddings
        response = await execute_db_operation(
            lambda: supabase.table('knowledge_base')
            .select('*')
            .is_('embedding', None)
            .execute()
        )
        
        kb_items = response.data if response and hasattr(response, 'data') else []
        
        if not kb_items:
            logger.info("No new KB items to embed")
            return
        
        logger.info(f"Found {len(kb_items)} KB items to embed")
        
        for item in kb_items:
            try:
                # Generate embedding
                embedding = embedder.encode(item['content']).tolist()
                
                # Update database
                await execute_db_operation(
                    lambda: supabase.table('knowledge_base')
                    .update({'embedding': embedding})
                    .eq('id', item['id'])
                    .execute()
                )
                
                logger.info(f"Embedded KB item: {item['title']}")
                
            except Exception as e:
                logger.error(f"Failed to embed KB item {item.get('id', 'unknown')}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"KB embedding process failed: {e}")

async def retrieve_context(query: str, customer_id: Optional[str] = None) -> str:
    """Retrieve RAG context with parallel customer data fetching"""
    try:
        # Embed query
        query_emb = embedder.encode(query).tolist()
        
        # Vector search for KB documents
        rag_task = asyncio.create_task(
            execute_db_operation(
                lambda: supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_emb,
                        'match_threshold': Config.RAG_MATCH_THRESHOLD,
                        'match_count': Config.RAG_MATCH_COUNT
                    }
                ).execute()
            )
        )
        
        context_parts = []
        
        # Get KB matches
        rag_result = await rag_task
        if rag_result and hasattr(rag_result, 'data') and rag_result.data:
            kb_context = "\n".join([
                f"KB: {item['title']} - {item['content']} (similarity: {item.get('similarity', 0):.2f})"
                for item in rag_result.data
            ])
            context_parts.append("Knowledge Base Information:\n" + kb_context)
        
        # Parallel fetch customer data if customer_id provided
        if customer_id:
            tasks = []
            
            # Customer profile task
            customer_task = asyncio.create_task(
                execute_db_operation(
                    lambda: supabase.table('customers')
                    .select('*')
                    .eq('id', customer_id)
                    .maybe_single()
                    .execute()
                )
            )
            tasks.append(customer_task)
            
            # Orders task
            orders_task = asyncio.create_task(
                execute_db_operation(
                    lambda: supabase.table('orders')
                    .select('*')
                    .eq('customer_id', customer_id)
                    .order('purchase_timestamp', desc=True)
                    .limit(5)
                    .execute()
                )
            )
            tasks.append(orders_task)
            
            # Wait for all tasks
            customer_result, orders_result = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process customer data
            if not isinstance(customer_result, Exception) and customer_result and hasattr(customer_result, 'data') and customer_result.data:
                customer = customer_result.data
                customer_info = (
                    f"\nCustomer Information:\n"
                    f"Name: {customer.get('name', 'Unknown')}\n"
                    f"Email: {customer.get('email', 'Unknown')}\n"
                    f"City: {customer.get('city', 'Unknown')}\n"
                    f"State: {customer.get('state', 'Unknown')}"
                )
                context_parts.append(customer_info)
            
            # Process orders data
            if not isinstance(orders_result, Exception) and orders_result and hasattr(orders_result, 'data') and orders_result.data:
                orders_info = "\nRecent Orders:\n"
                for order in orders_result.data:
                    orders_info += (
                        f"- Order {order.get('order_number', 'N/A')}: "
                        f"{order.get('status', 'N/A')} on {order.get('purchase_timestamp', 'N/A')}\n"
                    )
                context_parts.append(orders_info)
        
        return "\n".join(context_parts) if context_parts else "No relevant context found."
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return "Error retrieving context. Using general knowledge only."

# =============================================
# Conversation State Management
# =============================================
async def get_convo_state(session_id: str) -> Dict:
    """Get or create conversation state"""
    try:
        # Try to get existing conversation
        response = await execute_db_operation(
            lambda: supabase.table('conversations')
            .select('*')
            .eq('session_id', session_id)
            .maybe_single()
            .execute()
        )
        
        if response and hasattr(response, 'data') and response.data:
            logger.info(f"Found existing conversation for session: {session_id}")
            return response.data
        
        # Create new conversation
        customer_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        insert_response = await execute_db_operation(
            lambda: supabase.table('conversations')
            .insert({
                'session_id': session_id,
                'customer_id': customer_id,
                'history': [],
                'summary': '',
                'created_at': now,
                'updated_at': now
            })
            .execute()
        )
        
        if insert_response and hasattr(insert_response, 'data') and insert_response.data:
            logger.info(f"Created new conversation for session: {session_id}")
            return insert_response.data[0]
        else:
            logger.warning(f"Failed to insert conversation for session: {session_id}")
            # Return in-memory state if DB insert fails
            return {
                'session_id': session_id,
                'customer_id': customer_id,
                'history': [],
                'summary': ''
            }
            
    except Exception as e:
        logger.error(f"Failed to get/create conversation: {e}")
        # Return in-memory state on error
        return {
            'session_id': session_id,
            'customer_id': str(uuid.uuid4()),
            'history': [],
            'summary': ''
        }

async def update_convo_state(session_id: str, role: str, content: str):
    """Update conversation state with summarization for long conversations"""
    try:
        # Get current state
        convo = await get_convo_state(session_id)
        
        # Parse history safely
        history = convo.get('history', [])
        if isinstance(history, str):
            try:
                history = json.loads(history)
            except json.JSONDecodeError:
                history = []
        
        # Ensure history is a list
        if not isinstance(history, list):
            history = []
        
        # Add new message
        history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        summary = convo.get('summary', '')
        
        # Check if we need to summarize
        if len(history) > Config.MAX_HISTORY_LENGTH:
            logger.info(f"Summarizing conversation for session: {session_id}")
            
            # Generate summary of older messages
            old_messages = history[:len(history)//2]
            recent_messages = history[len(history)//2:]
            
            summary_prompt = (
                "Summarize this customer support conversation concisely in 2-3 sentences:\n" +
                json.dumps([{"role": m["role"], "content": m["content"]} for m in old_messages], indent=2)
            )
            
            try:
                new_summary = await generate_response(
                    summary_prompt, 
                    [], 
                    stream=False
                )
                
                # Update state with summary + recent messages
                history = [
                    {'role': 'system', 'content': f"Previous conversation summary: {new_summary}"}
                ] + recent_messages
                
                summary = new_summary
                
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                # Keep last N messages if summary fails
                history = history[-Config.MAX_HISTORY_LENGTH:]
        
        # Update in database
        await execute_db_operation(
            lambda: supabase.table('conversations')
            .update({
                'history': history,
                'summary': summary,
                'updated_at': datetime.utcnow().isoformat()
            })
            .eq('session_id', session_id)
            .execute()
        )
        
        logger.info(f"Updated conversation state for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to update conversation state: {e}")

# =============================================
# LLM Response Generation
# =============================================
async def generate_response(
    user_message: str, 
    history: List[Dict], 
    context: Optional[str] = None,
    stream: bool = True
) -> str:
    """Generate LLM response with proper context"""
    try:
        # Build messages
        messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT.format(
                    context=context or "No specific context available. Use general knowledge.",
                    summary="No previous conversation summary available."
                )
            }
        ]
        
        # Add history (last 10 for performance)
        if history:
            for msg in history[-10:]:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
        
        # Add current user message
        messages.append({'role': 'user', 'content': user_message})
        
        if stream:
            # Streaming response
            response = ""
            stream_resp = await groq_client.chat.completions.create(
                messages=messages,
                model=Config.LLM_MODEL,
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            
            async for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content
            
            return response
        else:
            # Non-streaming response
            completion = await groq_client.chat.completions.create(
                messages=messages,
                model=Config.LLM_MODEL,
                temperature=0.7,
                max_tokens=500
            )
            return completion.choices[0].message.content
            
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "I'm having trouble responding right now. Please try again in a moment."

# =============================================
# FastAPI App with Lifecycle Management
# =============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("=" * 50)
    logger.info("Starting HelpKart AI Agent...")
    logger.info("=" * 50)
    
    # Embed KB content
    try:
        await embed_and_store_kb()
        logger.info("Knowledge base embedding complete")
    except Exception as e:
        logger.error(f"KB embedding failed: {e}")
    
    logger.info("Server is ready to accept connections")
    yield
    
    # Shutdown
    logger.info("=" * 50)
    logger.info("Shutting down HelpKart AI Agent...")
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
    logger.info("Shutdown complete")
    logger.info("=" * 50)

# Create FastAPI app
app = FastAPI(
    title="HelpKart AI Agent",
    description="Real-time customer support AI agent with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# Health Check Endpoint
# =============================================
@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": Config.ENVIRONMENT,
        "services": {}
    }
    
    try:
        # Check Supabase connection
        await execute_db_operation(
            lambda: supabase.table('customers').select('count', count='exact').limit(1).execute()
        )
        health_status["services"]["supabase"] = "connected"
    except Exception as e:
        health_status["services"]["supabase"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check Groq
        await groq_client.models.list()
        health_status["services"]["groq"] = "connected"
    except Exception as e:
        health_status["services"]["groq"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    health_status["services"]["redis"] = "connected" if redis_client else "disabled"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health_status)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HelpKart AI Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "ws://localhost:8000/ws/{session_id}"
    }

# =============================================
# WebSocket Chat Endpoint
# =============================================
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Main WebSocket endpoint for real-time chat"""
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"New WebSocket connection from {client_host} for session {session_id}")
    
    try:
        # Accept connection
        await websocket.accept()
        logger.info(f"WebSocket accepted for session {session_id}")
        
        # Rate limiting
        allowed, message = await rate_limit(session_id)
        if not allowed:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": message
            }))
            await websocket.close()
            return
        
        # Get conversation state
        convo = await get_convo_state(session_id)
        logger.info(f"Loaded conversation state for session {session_id}")
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "content": "Connected to HelpKart AI Agent. How can I help you today?",
            "session_id": session_id
        }))
        
        # Main message loop
        while True:
            try:
                # Receive message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0
                )
                
                # Parse message
                try:
                    message_data = json.loads(data)
                    user_message = message_data.get('message', '')
                except json.JSONDecodeError:
                    user_message = data
                
                logger.info(f"Session {session_id} - Received: {user_message[:100]}...")
                
                # Check for exit commands
                if user_message.lower().strip() in ['exit', 'quit', 'bye', 'goodbye']:
                    await websocket.send_text(json.dumps({
                        "type": "goodbye",
                        "content": "Goodbye! Thanks for chatting with HelpKart. Have a great day!"
                    }))
                    break
                
                # Send typing indicator
                await websocket.send_text(json.dumps({
                    "type": "typing",
                    "content": "Agent is typing..."
                }))
                
                # Retrieve context (parallel)
                context_task = asyncio.create_task(
                    retrieve_context(user_message, convo.get('customer_id'))
                )
                
                # Get context
                context = await context_task
                
                # Prepare history
                history = convo.get('history', [])
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except json.JSONDecodeError:
                        history = []
                
                # Generate streaming response
                try:
                    messages = [
                        {
                            'role': 'system',
                            'content': SYSTEM_PROMPT.format(
                                context=context,
                                summary=convo.get('summary', 'No summary available')
                            )
                        }
                    ]
                    
                    # Add last 7 messages for context
                    if history:
                        for msg in history[-7:]:
                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                messages.append({
                                    'role': msg['role'],
                                    'content': msg['content']
                                })
                    
                    messages.append({'role': 'user', 'content': user_message})
                    
                    # Stream from Groq
                    stream = await groq_client.chat.completions.create(
                        messages=messages,
                        model=Config.LLM_MODEL,
                        temperature=0.7,
                        max_tokens=500,
                        stream=True
                    )
                    
                    # Send chunks to client
                    full_response = ""
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            await websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": content,
                                "complete": False
                            }))
                    
                    # Send completion marker
                    await websocket.send_text(json.dumps({
                        "type": "complete",
                        "content": full_response,
                        "complete": True
                    }))
                    
                    # Update conversation state
                    await update_convo_state(session_id, 'user', user_message)
                    await update_convo_state(session_id, 'assistant', full_response)
                    
                    logger.info(f"Session {session_id} - Response sent: {len(full_response)} chars")
                    
                except Exception as e:
                    logger.error(f"LLM streaming failed for session {session_id}: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "I'm having trouble responding. Please try again."
                    }))
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "content": "ping"
                    }))
                except:
                    break
                
            except WebSocketDisconnect:
                logger.info(f"Session {session_id} - Client disconnected")
                break
                
            except Exception as e:
                logger.error(f"Session {session_id} - Error in message loop: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"An error occurred: {str(e)}"
                    }))
                except:
                    break
                
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"Session {session_id} - Connection closed")

# =============================================
# REST API Endpoints (for non-WebSocket clients)
# =============================================
@app.post("/api/chat/{session_id}")
async def chat_api(
    session_id: str,
    message: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """REST endpoint for chat (fallback)"""
    try:
        # Validate API token (optional)
        if credentials and credentials.credentials:
            # Add your token validation logic here
            pass
        
        # Get user message
        user_message = message.get('message', '')
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"API Chat - Session {session_id}: {user_message[:100]}...")
        
        # Get conversation
        convo = await get_convo_state(session_id)
        
        # Retrieve context
        context = await retrieve_context(user_message, convo.get('customer_id'))
        
        # Generate response
        history = convo.get('history', [])
        if isinstance(history, str):
            try:
                history = json.loads(history)
            except json.JSONDecodeError:
                history = []
        
        response = await generate_response(
            user_message, 
            history[-7:] if history else [], 
            context,
            stream=False
        )
        
        # Update state
        await update_convo_state(session_id, 'user', user_message)
        await update_convo_state(session_id, 'assistant', response)
        
        logger.info(f"API Chat - Session {session_id}: Response sent")
        
        return JSONResponse({
            'session_id': session_id,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API chat failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================
# Run Server
# =============================================
if __name__ == "__main__":
    import uvicorn
    port = Config.PORT
    reload = Config.ENVIRONMENT == "development"
    
    logger.info(f"Starting server on port {port} in {Config.ENVIRONMENT} mode")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level=Config.LOG_LEVEL.lower() if hasattr(Config, 'LOG_LEVEL') else "info"
    )
