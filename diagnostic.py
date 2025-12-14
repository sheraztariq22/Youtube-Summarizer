"""
Diagnostic script to analyze the exact structure of fetched transcripts
"""

from youtube_transcript_api import YouTubeTranscriptApi

video_id = "nIonZ6-4nuU"  # Keras tutorial

print("=" * 70)
print("DIAGNOSTIC: Analyzing Transcript Structure")
print("=" * 70)

try:
    # Create API instance
    api = YouTubeTranscriptApi()
    
    # Get transcript list
    print(f"\n1️⃣ Fetching transcript for video: {video_id}")
    transcript_list = api.list(video_id)
    
    # Find English transcript
    transcript_obj = None
    for t in transcript_list:
        if t.language_code == 'en':
            transcript_obj = t
            break
    
    if not transcript_obj:
        print("❌ No English transcript found")
        exit(1)
    
    print(f"✅ Found English transcript")
    print(f"   Language: {transcript_obj.language_code}")
    print(f"   Auto-generated: {transcript_obj.is_generated}")
    
    # Fetch the transcript data
    print(f"\n2️⃣ Fetching transcript data...")
    transcript_data = transcript_obj.fetch()
    
    print(f"✅ Fetched transcript")
    print(f"   Type: {type(transcript_data)}")
    print(f"   Length: {len(transcript_data)}")
    
    # Analyze first item
    print(f"\n3️⃣ Analyzing first transcript item...")
    first_item = transcript_data[0]
    
    print(f"   Type: {type(first_item)}")
    print(f"   Class name: {first_item.__class__.__name__}")
    
    # Check if it's a dictionary
    print(f"\n   Is dict? {isinstance(first_item, dict)}")
    
    # List all attributes
    attributes = [attr for attr in dir(first_item) if not attr.startswith('_')]
    print(f"   Attributes: {attributes}")
    
    # Try different access methods
    print(f"\n4️⃣ Trying to access data...")
    
    # Method 1: Object attributes
    print(f"\n   Method 1: Object attributes")
    try:
        if hasattr(first_item, 'text'):
            print(f"   ✅ .text works: {first_item.text[:50]}")
        else:
            print(f"   ❌ .text doesn't exist")
    except Exception as e:
        print(f"   ❌ .text error: {e}")
    
    try:
        if hasattr(first_item, 'start'):
            print(f"   ✅ .start works: {first_item.start}")
        else:
            print(f"   ❌ .start doesn't exist")
    except Exception as e:
        print(f"   ❌ .start error: {e}")
    
    # Method 2: Dictionary access
    print(f"\n   Method 2: Dictionary access")
    try:
        text = first_item['text']
        print(f"   ✅ ['text'] works: {text[:50]}")
    except Exception as e:
        print(f"   ❌ ['text'] error: {e}")
    
    try:
        start = first_item['start']
        print(f"   ✅ ['start'] works: {start}")
    except Exception as e:
        print(f"   ❌ ['start'] error: {e}")
    
    # Show raw representation
    print(f"\n5️⃣ Raw representation:")
    print(f"   {repr(first_item)}")
    
    # Try to convert to string
    print(f"\n6️⃣ String conversion:")
    print(f"   str(): {str(first_item)}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()