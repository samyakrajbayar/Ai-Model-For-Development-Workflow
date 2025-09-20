"""
Reddit-Style Shorts Generator
============================

Setup Instructions:
pip install streamlit moviepy gtts openai
mkdir backgrounds
# Place some MP4 background loops inside ./backgrounds
export OPENAI_API_KEY="your_api_key_here"  # Only needed for OpenAI TTS
streamlit run app.py

This app converts text scripts into engaging short videos with:
- TTS narration (Google TTS or OpenAI TTS)
- Random background videos
- Word-by-word synchronized captions
"""

import streamlit as st
import os
import random
import tempfile
import re
from pathlib import Path
import time

# Import required libraries
try:
    from moviepy.editor import (
        VideoFileClip, 
        CompositeVideoClip, 
        TextClip, 
        AudioFileClip,
        concatenate_videoclips
    )
    from gtts import gTTS
    import openai
except ImportError as e:
    st.error(f"Missing required library: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Reddit Shorts Generator",
    page_icon="🎬",
    layout="wide"
)

class RedditShortsGenerator:
    def __init__(self):
        self.backgrounds_dir = Path("backgrounds")
        self.temp_dir = Path(tempfile.gettempdir()) / "reddit_shorts"
        self.temp_dir.mkdir(exist_ok=True)
        
    def get_background_videos(self):
        """Get list of available background videos"""
        if not self.backgrounds_dir.exists():
            return []
        
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(list(self.backgrounds_dir.glob(f'*{ext}')))
        return videos
    
    def generate_tts_audio(self, text, engine="google", voice="alloy"):
        """Generate TTS audio using selected engine"""
        audio_file = self.temp_dir / f"narration_{int(time.time())}.mp3"
        
        if engine == "google":
            # Use Google TTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(audio_file))
            
        elif engine == "openai":
            # Use OpenAI TTS
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            client = openai.OpenAI(api_key=api_key)
            
            try:
                response = client.audio.speech.create(
                    model="tts-1",  # Using tts-1 as it's more widely available
                    voice=voice,
                    input=text
                )
                response.stream_to_file(str(audio_file))
            except Exception as e:
                raise ValueError(f"OpenAI TTS error: {str(e)}")
        
        return str(audio_file)
    
    def estimate_word_timing(self, text, audio_duration):
        """Estimate timing for each word based on audio duration"""
        words = text.split()
        if not words:
            return []
        
        # Simple estimation: divide duration by number of words
        # Add some variation for more natural timing
        base_duration_per_word = audio_duration / len(words)
        
        word_timings = []
        current_time = 0
        
        for i, word in enumerate(words):
            # Adjust timing based on word length and punctuation
            word_factor = 1.0
            if len(word) > 8:  # Longer words take more time
                word_factor = 1.3
            elif len(word) < 3:  # Shorter words are quicker
                word_factor = 0.7
            
            if word.endswith(('.', '!', '?', ',')):  # Punctuation adds pause
                word_factor += 0.3
            
            duration = base_duration_per_word * word_factor
            word_timings.append({
                'word': word,
                'start': current_time,
                'end': current_time + duration
            })
            current_time += duration
        
        return word_timings
    
    def create_caption_clips(self, word_timings, video_size):
        """Create individual text clips for each word"""
        caption_clips = []
        
        for timing in word_timings:
            # Create text clip for each word
            txt_clip = TextClip(
                timing['word'],
                fontsize=60,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=3
            ).set_position(('center', video_size[1] * 0.85)).set_start(
                timing['start']
            ).set_duration(
                timing['end'] - timing['start']
            )
            
            caption_clips.append(txt_clip)
        
        return caption_clips
    
    def generate_short_video(self, script, tts_engine, voice="alloy", progress_callback=None):
        """Generate the complete short video"""
        try:
            # Update progress
            if progress_callback:
                progress_callback("Selecting background video...")
            
            # Get random background video
            background_videos = self.get_background_videos()
            if not background_videos:
                raise ValueError("No background videos found in 'backgrounds/' folder")
            
            background_path = random.choice(background_videos)
            
            # Update progress
            if progress_callback:
                progress_callback("Generating TTS audio...")
            
            # Generate TTS audio
            audio_path = self.generate_tts_audio(script, tts_engine, voice)
            
            # Update progress
            if progress_callback:
                progress_callback("Processing video and audio...")
            
            # Load audio to get duration
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # Load and prepare background video
            background_clip = VideoFileClip(str(background_path))
            
            # Loop or trim background to match audio duration
            if background_clip.duration < audio_duration:
                # Loop the background video
                loops_needed = int(audio_duration / background_clip.duration) + 1
                background_clip = concatenate_videoclips([background_clip] * loops_needed)
            
            # Trim to exact duration
            background_clip = background_clip.subclip(0, audio_duration)
            
            # Add audio to background
            background_clip = background_clip.set_audio(audio_clip)
            
            # Update progress
            if progress_callback:
                progress_callback("Creating word-by-word captions...")
            
            # Create word timings
            word_timings = self.estimate_word_timing(script, audio_duration)
            
            # Create caption clips
            caption_clips = self.create_caption_clips(word_timings, background_clip.size)
            
            # Update progress
            if progress_callback:
                progress_callback("Compositing final video...")
            
            # Composite everything together
            final_video = CompositeVideoClip([background_clip] + caption_clips)
            
            # Export final video
            output_path = self.temp_dir / f"reddit_short_{int(time.time())}.mp4"
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / 'temp-audio.m4a'),
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio_clip.close()
            background_clip.close()
            final_video.close()
            for clip in caption_clips:
                clip.close()
            
            # Remove temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return str(output_path)
            
        except Exception as e:
            raise Exception(f"Error generating video: {str(e)}")

# Initialize the generator
@st.cache_resource
def get_generator():
    return RedditShortsGenerator()

# Main Streamlit app
def main():
    st.title("🎬 Reddit-Style Shorts Generator")
    st.markdown("Transform your text scripts into engaging short videos with TTS narration and captions!")
    
    generator = get_generator()
    
    # Check for background videos
    background_videos = generator.get_background_videos()
    if not background_videos:
        st.warning("⚠️ No background videos found!")
        st.info("""
        Please add MP4 background videos to the `backgrounds/` folder.
        
        Suggested background types:
        - Soap cutting loops
        - Paint mixing videos
        - Slime stretching
        - Satisfying loop animations
        """)
        return
    
    st.success(f"✅ Found {len(background_videos)} background videos")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Script Input")
        script = st.text_area(
            "Enter your script:",
            height=200,
            placeholder="Enter your text here... This will be converted to speech and displayed as captions.",
            help="Write engaging content that works well for short videos (30-60 seconds recommended)"
        )
    
    with col2:
        st.subheader("🎙️ TTS Settings")
        
        # TTS Engine selection
        tts_engine = st.selectbox(
            "TTS Engine:",
            ["google", "openai"],
            format_func=lambda x: "Google TTS (Free)" if x == "google" else "OpenAI TTS (API Key Required)"
        )
        
        # Voice selection for OpenAI
        voice = "alloy"  # Default voice
        if tts_engine == "openai":
            voice = st.selectbox(
                "Voice:",
                ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                help="Different voice personalities for OpenAI TTS"
            )
            
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OPENAI_API_KEY environment variable not set")
    
    # Generate button
    st.markdown("---")
    
    if st.button("🎬 Generate Short Video", type="primary", use_container_width=True):
        if not script.strip():
            st.error("Please enter a script!")
            return
        
        if tts_engine == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key is required for OpenAI TTS!")
            return
        
        # Progress tracking
        progress_container = st.empty()
        
        try:
            with st.spinner("Generating your short video..."):
                def update_progress(message):
                    progress_container.info(f"🔄 {message}")
                
                # Generate the video
                output_path = generator.generate_short_video(
                    script=script,
                    tts_engine=tts_engine,
                    voice=voice,
                    progress_callback=update_progress
                )
                
                progress_container.success("✅ Video generated successfully!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
        
        # Display results
        st.markdown("---")
        st.subheader("🎉 Your Generated Short Video")
        
        # Display video
        if os.path.exists(output_path):
            st.video(output_path)
            
            # Download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download MP4",
                    data=file.read(),
                    file_name=f"reddit_short_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        
        # Video info
        st.info(f"""
        **Video Details:**
        - TTS Engine: {tts_engine.title()}
        - Voice: {voice if tts_engine == 'openai' else 'Default'}
        - Background: Random from library
        - Captions: Word-by-word synchronized
        """)

if __name__ == "__main__":
    main()
