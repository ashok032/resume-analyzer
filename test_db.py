from db import get_connection

def test_connection():
    try:
        conn = get_connection()
        print("✅ Connected to DB successfully:", conn)
        conn.close()
    except Exception as e:
        print("❌ Connection failed:", e)

if __name__ == "__main__":
    test_connection()
