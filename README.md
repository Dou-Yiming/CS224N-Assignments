# CS224N-Assignments
The assignments of Stanford CS224N: NLP with Deep Learning | Winter 2019

### Result of each assignment:

- **Assignment1: Exploring Word Vectors**

    Unfinished due to limited RAM space

- **Assignment2: word2vec**

    <p align="center">
    <img src="./Assignment2/output/word_vectors.png" width="400"/>
    </p>

- **Assignment3: Dependency Parsing**
  
    - The code is modified to support training on **CUDA**, about **30x** faster!
    - Result:
        - Average Train Loss: 0.018
        - best dev UAS: 88.32
        - test UAS: 89.06
    
- **Assignment4: Neural Machine Translation with RNNs**

    - Corpus BLEU: 21.49

    - Translation example:

        - origin text (Spanish): 

            ```
            Necesita verdad y belleza, y estoy muy felz que hoy se habl mucho de esto.
            Tambin necesita -- necesita dignidad, amor y placer. Y es nuestro trabajo proporcionar esas cosas.
            Gracias
            ```

        - translated text (English):

            ```
            It needs truth and beauty and I'm very happy that today I was talking about this.
            It also needs -- needs dignity, love and pleasure. And it's our job to provide those things.
            Thank you.
            ```

            
