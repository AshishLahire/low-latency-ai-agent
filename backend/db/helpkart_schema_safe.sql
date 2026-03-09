-- ============================================================
-- HelpKart Schema — SAFE VERSION (re-runnable)
-- Uses IF NOT EXISTS everywhere — safe to run multiple times
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. CUSTOMERS
CREATE TABLE IF NOT EXISTS customers (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    phone         TEXT,
    tier          TEXT NOT NULL DEFAULT 'standard' CHECK (tier IN ('standard', 'premium', 'vip')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);

-- 2. ORDERS
CREATE TABLE IF NOT EXISTS orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     UUID NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled','refunded')),
    total_amount    NUMERIC(10, 2) NOT NULL CHECK (total_amount >= 0),
    currency        TEXT NOT NULL DEFAULT 'INR',
    items           JSONB NOT NULL DEFAULT '[]',
    tracking_number TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status      ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at  ON orders(created_at DESC);

-- 3. KNOWLEDGE BASE
CREATE TABLE IF NOT EXISTS knowledge_base (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL,
    embedding   VECTOR(384),
    metadata    JSONB NOT NULL DEFAULT '{}',
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kb_category  ON knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_kb_is_active ON knowledge_base(is_active);

-- IVFFlat index only if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE indexname = 'idx_kb_embedding'
  ) THEN
    CREATE INDEX idx_kb_embedding ON knowledge_base
      USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  END IF;
END $$;

-- 4. SESSIONS
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     UUID REFERENCES customers(id) ON DELETE SET NULL,
    channel         TEXT NOT NULL DEFAULT 'text' CHECK (channel IN ('text', 'voice')),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed', 'abandoned')),
    context_window  JSONB NOT NULL DEFAULT '[]',
    context_summary TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sessions_customer_id ON sessions(customer_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status      ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active_at DESC);

-- 5. MESSAGES
CREATE TABLE IF NOT EXISTS messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content          TEXT NOT NULL,
    retrieved_chunks JSONB DEFAULT '[]',
    latency_ms       INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, created_at ASC);

-- 6. TRIGGER FUNCTION
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- Triggers (drop first so re-running doesn't error)
DROP TRIGGER IF EXISTS trg_customers_updated_at   ON customers;
DROP TRIGGER IF EXISTS trg_orders_updated_at      ON orders;
DROP TRIGGER IF EXISTS trg_kb_updated_at          ON knowledge_base;

CREATE TRIGGER trg_customers_updated_at
    BEFORE UPDATE ON customers FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_orders_updated_at
    BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_kb_updated_at
    BEFORE UPDATE ON knowledge_base FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- 7. RLS
ALTER TABLE customers      ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders         ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages       ENABLE ROW LEVEL SECURITY;

-- 8. SEED DATA (only insert if table is empty)
INSERT INTO customers (id, name, email, phone, tier)
SELECT * FROM (VALUES
    ('11111111-1111-1111-1111-111111111111'::uuid, 'Priya Sharma',  'priya@example.com',  '+91-9876543210', 'premium'),
    ('22222222-2222-2222-2222-222222222222'::uuid, 'Arjun Mehta',   'arjun@example.com',  '+91-9123456789', 'standard'),
    ('33333333-3333-3333-3333-333333333333'::uuid, 'Sunita Rao',    'sunita@example.com', '+91-9988776655', 'vip')
) AS v(id, name, email, phone, tier)
WHERE NOT EXISTS (SELECT 1 FROM customers LIMIT 1);

INSERT INTO orders (customer_id, status, total_amount, items, tracking_number)
SELECT * FROM (VALUES
    ('11111111-1111-1111-1111-111111111111'::uuid, 'shipped',   1299.00::numeric, '[{"name":"Wireless Earbuds","qty":1,"price":1299}]'::jsonb, 'HK-TRK-001'),
    ('11111111-1111-1111-1111-111111111111'::uuid, 'delivered',  499.00::numeric, '[{"name":"Phone Case","qty":2,"price":249.50}]'::jsonb,    'HK-TRK-002'),
    ('22222222-2222-2222-2222-222222222222'::uuid, 'pending',   3599.00::numeric, '[{"name":"Bluetooth Speaker","qty":1,"price":3599}]'::jsonb, NULL),
    ('33333333-3333-3333-3333-333333333333'::uuid, 'cancelled',  899.00::numeric, '[{"name":"USB-C Hub","qty":1,"price":899}]'::jsonb,          NULL)
) AS v(customer_id, status, total_amount, items, tracking_number)
WHERE NOT EXISTS (SELECT 1 FROM orders LIMIT 1);

INSERT INTO knowledge_base (title, category, content)
SELECT * FROM (VALUES
    ('Return Policy',        'returns',  'HelpKart accepts returns within 30 days of delivery. Items must be unused and in original packaging. Refunds are processed within 5-7 business days after the returned item is received.'),
    ('Shipping Information', 'shipping', 'Standard shipping takes 3-5 business days. Express shipping (1-2 days) is available for an additional fee. Free standard shipping on orders above ₹999.'),
    ('Track Your Order',     'shipping', 'You can track your order using the tracking number sent to your email. Visit helpkart.com/track or ask our support agent with your order ID.'),
    ('Payment Methods',      'payment',  'HelpKart accepts UPI, credit/debit cards, net banking, and cash on delivery. EMI options are available on orders above ₹3000.'),
    ('Cancel an Order',      'orders',   'Orders can be cancelled within 1 hour of placement if not yet confirmed. After confirmation, cancellation may not be possible. Contact support immediately for urgent cancellations.'),
    ('Account & Password',   'account',  'To reset your password, visit helpkart.com/forgot-password and enter your registered email. A reset link will be sent within 2 minutes.')
) AS v(title, category, content)
WHERE NOT EXISTS (SELECT 1 FROM knowledge_base LIMIT 1);
