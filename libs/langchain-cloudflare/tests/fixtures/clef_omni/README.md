# Clef Omni media fixtures

These synthetic fixtures contain only the statements and animations below.
Tests read the committed bytes; speech synthesis and FFmpeg are not test dependencies.

| File | Content |
| --- | --- |
| `door_open.wav` | “The door is open.” |
| `door_closed.wav` | “The door is closed.” |
| `square.mp4` | A red square on a white background; no audio. |
| `circle.mp4` | A red circle on a white background; no audio. |
| `square_open.mp4` | The square animation with the open-door recording. |
| `circle_closed.mp4` | The circle animation with the closed-door recording. |

Audio was synthesized with macOS `say`, the Samantha voice, at 140 words per
minute, as mono 16-bit PCM WAV at 16 kHz. Video contains 30 RGB frames at
160×120 pixels and 10 fps, encoded with FFmpeg's `libx264` encoder, CRF 24,
and `yuv420p` pixel format. The square is 60×60 pixels at y=30, with its
left edge moving from x=95 to x=5. The circle has radius 30 pixels and a
center moving from (34.5, 59.5) to (124.5, 59.5). Each animation holds its
starting position for the first ten frames, moves linearly over the middle
ten frames, and holds its final position for the last ten frames. Soundtracks
are AAC at 32 kbit/s, padded to the three-second video length. Container
metadata was stripped with `-map_metadata -1`.

The same media bytes are tested as data URLs and as `content_type`/`base64`
objects through synchronous REST, asynchronous REST, and Python Worker bindings.
The mixed-media case uses the existing red PNG fixture, the open-door audio,
and the silent circle video. Prompts do not reveal the expected answers.

Acceptance checks distinguish square from circle and open-door speech from
closed-door speech, including video soundtracks. They require correct native
answers, valid probability distributions, and token usage. Motion-direction
accuracy is a separate exploratory model-quality check.
