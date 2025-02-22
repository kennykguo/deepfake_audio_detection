Pyannnote is an open source toolkit written in Python for speaker diarization. The toolkit is built on the PyTorch machine learning framework and provides a set of trainable neural building blocks that can be combined and optimized to create speaker diarization pipelines

Speaker diaratization - partitioning a audio stream into temporal segments based on the identity of the speaker - activity, change, overlap


The toolkit offers modular components for tasks like voice activity detection (VAD), speaker change detection, overlap detection, and speaker embedding.

Also comes with speech embeddings - Pyannote implements advanced metric learning techniques (e.g., additive angular margin loss, contrastive loss) for speaker embeddings. The toolkit provides tools for grouping speech segments by speaker identity.


Setup. If using this repo, must cite it
https://github.com/pyannote/pyannote-audio