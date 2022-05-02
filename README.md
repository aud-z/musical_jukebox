# Mood2Music: Art & Machine Learning Project 4 (Spring 2022)

This is the repostiory for Mood2Music: a music generation process that generates both style, lyrics, and instrumentals in a song that matches your mood. 

- The lyric generation is done using GPT-3's Curie model adapted from the paper _"Language Models are Few-Shot Learners"_ [[1]](#Citation), and is fine-tuned 
on a user's Spotify liked songs. 
- The artist and genre selection is done through a clustering algorithm using an "Akinator" style question/answer system of song preferences.
- The music generation is done using OpenAI's Jukebox algorithm, adapted from the paper _"Jukebox: A Generative Model for Music"_ [[2]](#Citation).

## Citations

[1] Brown, Tom B. et al. "Language Models are Few-Shot Learners." (2020).

[2] Dhariwal, Prafulla et al. "Jukebox: A Generative Model for Music." (2020).
<img width="850" alt="Screen Shot 2022-05-02 at 1 15 10 AM" src="https://user-images.githubusercontent.com/10016632/166187439-c0715dea-87bb-4207-ae7a-3ffa89bb0146.png">
