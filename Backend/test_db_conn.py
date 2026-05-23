import asyncio
from app.database import engine
from sqlalchemy import text

async def test_connection():
    from app.config import settings
    # Mask password for safety
    masked_url = settings.DATABASE_URL
    if ":" in masked_url and "@" in masked_url:
        parts = masked_url.split("@")
        user_pass = parts[0].split(":")
        if len(user_pass) > 2:
            user_pass[2] = "*****"
            masked_url = ":".join(user_pass) + "@" + "@".join(parts[1:])
    print(f"Testing connection using URL: {masked_url}")
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version();"))
            row = result.fetchone()
            print("Connection successful! [OK]")
            print("PostgreSQL Version:", row[0])
    except Exception as e:
        print("Connection failed! [ERROR]")
        print("Error details:", e)
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
