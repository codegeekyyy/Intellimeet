import asyncio
import asyncpg

async def main():
    print("Testing connection with asyncpg directly...")
    
    # Try localhost
    try:
        print("\n1. Trying to connect to localhost...")
        conn = await asyncpg.connect(
            user='postgres',
            password='Deep@2003',
            database='intellimeet',
            host='localhost'
        )
        print("Success on localhost!")
        await conn.close()
    except Exception as e:
        print(f"Failed on localhost: {e}")
        
    # Try 127.0.0.1 (IPv4 loopback)
    try:
        print("\n2. Trying to connect to 127.0.0.1...")
        conn = await asyncpg.connect(
            user='postgres',
            password='Deep@2003',
            database='intellimeet',
            host='127.0.0.1'
        )
        print("Success on 127.0.0.1!")
        await conn.close()
    except Exception as e:
        print(f"Failed on 127.0.0.1: {e}")

    # Try ::1 (IPv6 loopback)
    try:
        print("\n3. Trying to connect to ::1...")
        conn = await asyncpg.connect(
            user='postgres',
            password='Deep@2003',
            database='intellimeet',
            host='::1'
        )
        print("Success on ::1!")
        await conn.close()
    except Exception as e:
        print(f"Failed on ::1: {e}")

if __name__ == '__main__':
    asyncio.run(main())
