import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_USER = "vtsk"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "vtsk_db"

def create_database():
    print("Connecting to PostgreSQL...")
    
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if exists:
            print(f"Database '{DB_NAME}' already exists")
        else:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully")
        
        cursor.close()
        conn.close()
        
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
        print("Creating tables...")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_sessions (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(255),
                attack_type VARCHAR(50),
                started_at TIMESTAMP DEFAULT NOW(),
                ended_at TIMESTAMP,
                status VARCHAR(20) DEFAULT 'running',
                total_requests INTEGER DEFAULT 0,
                requests_sent INTEGER DEFAULT 0,
                requests_received INTEGER DEFAULT 0,
                requests_blocked INTEGER DEFAULT 0,
                avg_response_time FLOAT,
                min_response_time FLOAT,
                max_response_time FLOAT,
                throughput_rps FLOAT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_requests (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(50) UNIQUE NOT NULL,
                batch_id VARCHAR(50),
                attack_type VARCHAR(30) DEFAULT 'normal',
                payload_size INTEGER DEFAULT 0,
                sent_at TIMESTAMP NOT NULL,
                source_ip VARCHAR(45),
                target_endpoint VARCHAR(255),
                http_method VARCHAR(10) DEFAULT 'POST',
                headers_count INTEGER DEFAULT 0,
                is_malicious BOOLEAN DEFAULT FALSE,
                malicious_pattern VARCHAR(50)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_responses (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(50) NOT NULL,
                batch_id VARCHAR(50),
                received_at TIMESTAMP NOT NULL,
                response_time_ms FLOAT,
                status_code INTEGER,
                was_blocked BOOLEAN DEFAULT FALSE,
                blocked_by VARCHAR(50),
                source_ip VARCHAR(45),
                passed_through BOOLEAN DEFAULT TRUE,
                error_message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocked_requests (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(50) NOT NULL,
                session_id VARCHAR(50),
                blocked_at TIMESTAMP DEFAULT NOW(),
                blocked_by VARCHAR(50),
                block_reason TEXT,
                source_ip VARCHAR(45),
                attack_signature VARCHAR(100)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latency_metrics (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(50),
                timestamp TIMESTAMP DEFAULT NOW(),
                interval_seconds INTEGER DEFAULT 1,
                requests_count INTEGER DEFAULT 0,
                avg_latency_ms FLOAT,
                p50_latency_ms FLOAT,
                p95_latency_ms FLOAT,
                p99_latency_ms FLOAT,
                errors_count INTEGER DEFAULT 0,
                blocked_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protection_events (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(50),
                event_type VARCHAR(50),
                source VARCHAR(50),
                timestamp TIMESTAMP DEFAULT NOW(),
                details TEXT,
                severity VARCHAR(20),
                source_ip VARCHAR(45),
                action_taken VARCHAR(50)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_sent ON traffic_requests(batch_id, sent_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_batch ON traffic_responses(batch_id, received_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON traffic_responses(request_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON test_sessions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_session ON blocked_requests(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON protection_events(session_id)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("All tables created successfully!")
        print(f"\nDatabase ready: {DB_NAME}")
        print(f"Connection: postgresql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
    except psycopg2.OperationalError as e:
        print(f"\nERROR: Cannot connect to PostgreSQL!")
        print(f"Details: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  1. Check if PostgreSQL service is started")
        print("  2. Verify credentials in .env file")
        print(f"  3. Ensure user '{DB_USER}' exists with password '{DB_PASSWORD}'")
        print(f"  4. PostgreSQL should be listening on {DB_HOST}:{DB_PORT}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Database Setup Script")
    print("=" * 50)
    success = create_database()
    if success:
        print("\nYou can now run:")
        print("  python run_receiver.py")
        print("  python run_sender.py")
