# Check status
curl http://localhost:8000/status

# Generate video
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "A serene koi pond at night with glowing lanterns",
       "width": 720,
       "height": 480,
       "num_frames": 41,
       "num_inference_steps": 20
     }' \
     --output generated_video.mp4