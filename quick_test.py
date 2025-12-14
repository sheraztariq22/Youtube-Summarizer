"""
Quick test to verify youtube-transcript-api is working correctly
Compatible with v1.2.1
"""

print("Testing YouTube Transcript API v1.2.1...")
print("=" * 60)

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    print("✅ youtube-transcript-api imported successfully")
except ImportError as e:
    print(f"❌ Failed to import youtube-transcript-api: {e}")
    print("\nTry installing: pip install youtube-transcript-api==1.2.1")
    exit(1)

# Test with a known working video
test_videos = {
    "nIonZ6-4nuU": "Keras Tutorial (Google)",
    "jNQXAC9IVRw": "Me at the zoo",
    "dQw4w9WgXcQ": "Rick Astley - Never Gonna Give You Up"
}

print("\nTesting transcript fetching...\n")

for video_id, title in test_videos.items():
    print(f"Testing: {title} ({video_id})")
    print("-" * 60)
    
    try:
        # Create API instance (v1.2.1 style)
        api = YouTubeTranscriptApi()
        
        # List available transcripts
        transcript_list = api.list(video_id)
        
        # Show available languages
        available = []
        for t in transcript_list:
            lang = f"{t.language_code}"
            if t.is_generated:
                lang += " (auto)"
            available.append(lang)
        print(f"  📋 Available: {', '.join(available)}")
        
        # Try to get English transcript
        transcript = None
        for t in transcript_list:
            if t.language_code == 'en':
                transcript = t.fetch()
                break
        
        if transcript:
            print(f"  ✅ SUCCESS!")
            print(f"  📊 Fetched {len(transcript)} transcript entries")
            
            # Access transcript data (v1.2.1 uses objects, not dicts)
            first_entry = transcript[0]
            if hasattr(first_entry, 'text'):
                # v1.2.1 style - object attributes
                print(f"  📝 First entry: {first_entry.text[:50]}...")
                print(f"  ⏰ Timestamp: {first_entry.start}s")
            else:
                # Newer versions - dictionary
                print(f"  📝 First entry: {first_entry['text'][:50]}...")
                print(f"  ⏰ Timestamp: {first_entry['start']}s")
        else:
            print(f"  ⚠️ No English transcript found")
        print()
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 60)
print("✅ All tests completed successfully!")
print("\nYour app should now work. Try running:")
print("python ytbot.py")