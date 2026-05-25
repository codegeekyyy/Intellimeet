import asyncio
import os
import uuid
import httpx
from app.main import app
from app.services.storage import get_audio_full_path

async def test_meeting_lifecycle():
    # 1. Create a dummy WAV file for testing
    dummy_wav_name = "test_audio.wav"
    # 44-byte standard empty WAV header
    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    with open(dummy_wav_name, "wb") as f:
        f.write(wav_header)

    print("\n--- Testing Meeting & Ingestion Flow (Async) ---")
    
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"meeting_user_{random_suffix}@example.com"
    test_username = f"meeting_user_{random_suffix}"
    test_password = "securepassword123"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # A. Register & Login
        print("Registering test user...")
        reg_res = await client.post("/auth/register", json={
            "email": test_email, "username": test_username, "password": test_password
        })
        assert reg_res.status_code == 201
        
        print("Logging in to get access token...")
        login_res = await client.post("/auth/login", json={
            "email": test_email, "password": test_password
        })
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # B. Test Upload Endpoint
        print("Testing POST /audio/upload...")
        with open(dummy_wav_name, "rb") as f:
            upload_res = await client.post(
                "/audio/upload",
                headers=headers,
                files={"file": (dummy_wav_name, f, "audio/wav")},
                data={"title": "Weekly Sync Meeting"}
            )
        
        assert upload_res.status_code == 201, f"Upload failed: {upload_res.text}"
        res_json = upload_res.json()
        assert "session_id" in res_json
        assert "job_id" in res_json
        assert res_json["status"] == "queued"
        session_id = res_json["session_id"]
        print(f"File uploaded successfully! Created session: {session_id}")

        # C. Test List Meetings Endpoint
        print("Testing GET /meetings/...")
        list_res = await client.get("/meetings/", headers=headers)
        assert list_res.status_code == 200
        meetings_list = list_res.json()
        assert len(meetings_list) >= 1
        assert meetings_list[0]["title"] == "Weekly Sync Meeting"
        print("Meetings listed successfully! [OK]")

        # D. Test Meeting Detail Endpoint
        print(f"Testing GET /meetings/{session_id}...")
        detail_res = await client.get(f"/meetings/{session_id}", headers=headers)
        assert detail_res.status_code == 200
        meeting_detail = detail_res.json()
        assert meeting_detail["title"] == "Weekly Sync Meeting"
        assert meeting_detail["status"] == "queued"
        print("Meeting details retrieved successfully! [OK]")

        # E. Test Deletion Endpoint
        print(f"Testing DELETE /meetings/{session_id}...")
        delete_res = await client.delete(f"/meetings/{session_id}", headers=headers)
        assert delete_res.status_code == 200
        print("Meeting deleted in DB successfully! [OK]")

        # F. Check that local file is cleaned up on disk
        print("Verifying audio file cleanup from disk...")
        # Since it is a relative path stored under settings.UPLOAD_DIR/{user_id}/{uuid}.ext,
        # let's verify it got deleted. We'll search the user's upload directory.
        user_upload_dir = os.path.join("./uploads", str(reg_res.json()["id"]))
        # It should be empty or not exist
        if os.path.exists(user_upload_dir):
            files = os.listdir(user_upload_dir)
            assert len(files) == 0, f"Local files were not cleaned up: {files}"
        print("Local audio file deleted from disk successfully! [OK]")

    # Cleanup the local dummy file we created at start
    if os.path.exists(dummy_wav_name):
        os.remove(dummy_wav_name)
        
    print("--- All Meeting & Ingestion Flow Tests Passed! ---")

if __name__ == "__main__":
    asyncio.run(test_meeting_lifecycle())
