
# Contest 3 Report
(This readme is actually our report of the contest, and the codes in the jupyter notebooks are just for reviewing)

Team Members:
- 103062240 蔡宗宇
- 103062224 劉哲宇
- 103062101 陳孜宇

----

## What we have done

Before any improvement to make, we first ran the given sample code for 200 epochs and submit the evaluation result to kaggle, expecting that our model will not perform well.

Surprisingly, our public score reached 2.34422, which is way higher than TA-80 (3.01099).

Because of the high score we got, we added some changes based on the model trained above to improve its performance:

- Gradient clipping
- Beam search

### Gradient Clipping
Like the sample code:

1. Compute the gradient of trainable variables.
2. Clip the gradient by tf.clip_by_norm.
3. Apply the clipped gradient to trainable vairables.

Gradient is easy to implement and apply to resolve the problem of gradient explosion.

However, the model of sample code version can somehow be trained pretty well and inferences reasonable captions, so we don't really feel that gradient clipping gives us great improvement on training stage.


```python
optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
                
grads_and_vars = optimizer.compute_gradients(
    self.total_loss, tf.trainable_variables())

clipped_grads_and_vars = [
    (tf.clip_by_norm(gv[0], 1.0), gv[1]) for gv in grads_and_vars]

self.train_op = optimizer.apply_gradients(
    clipped_grads_and_vars, global_step=self.global_step)
```

### Beam search

Beam search is a search algorithm that works well when computation of searching all nodes in graph is expensive, which fit the condition of our inference stage.

Beam search uses BFS to build the search tree, but it only use the top-k (user defined) best states at each searching stage. So with beam search, we can find many possible/well predictions and pick the best from the candidates according to the score of each.<br>
<img src='./beam.jpg'><br>

Since we don't have a 'score function' to calculate the joint probability of those partial captions generated when searching, we simply use the softmax of logits output by rnn_cell as the joint probability.

The following code is how we implement beam search in ImageCaptionModel.


```python
def beam_search(self, sess, rnn_state, prev_word, log_beam_prob, beam_size=3):
    probs, next_state = sess.run(
        fetches=['optimize/softmax:0', 'rnn_scope/final_state:0'],
        feed_dict={
            'input_feed:0': [prev_word],
            'rnn_scope/state_feed:0': rnn_state
        })
    probs = probs[0]
    probs_logsum = np.log(probs + 0.00001) + log_beam_prob
    indices = np.argsort(probs_logsum)[::-1][0:beam_size]
    best_probs = []
    for idx in indices:
        best_probs.append(probs_logsum[idx])
    next_beam_probs = []
    next_words = []
    for i in range(beam_size):
        next_beam_probs.append(best_probs[i])
        next_words.append(indices[i])
    return next_state, next_words, next_beam_probs

def beam_inference(self, sess, img_embed, enc_map, dec_map):
    st, ed = enc_map['<ST>'], enc_map['<ED>']
    initial_state = sess.run(
        fetches='rnn_scope/initial_state:0',
        feed_dict={'image_embed:0': img_embed})
    start_word_feed = st
    beam_size = 3
    state, words, probs = self.beam_search(sess, initial_state, start_word_feed, [0], beam_size)
    states = [state for i in range(beam_size)]
    captions = [[] for i in range(beam_size)]
    for i in range(beam_size):
        captions[i].append(words[i])
    for i in range(self.hps.max_caption_len - 1):
        all_beam_states = []
        all_beam_words = []
        all_beam_probs = []
        for j in range(beam_size):
            nstate, nwords, nprobs = self.beam_search(sess, states[j], words[j], probs[j], beam_size)
            for _ in range(beam_size):
                all_beam_states.append(nstate)
            all_beam_words.extend(nwords)
            all_beam_probs.extend(nprobs)
        indices = (np.argsort(all_beam_probs)[::-1])[0:beam_size]
        new_captions = [[] for i in range(beam_size)]
        for j, index in enumerate(indices):
            cap_id = index // beam_size
            new_captions[j].extend(captions[cap_id])
            new_captions[j].append(all_beam_words[index])
            states[j] = all_beam_states[index]
            words[j] = all_beam_words[index]
            probs[j] = all_beam_probs[index]
        captions = new_captions
    caption_sentences = []
    for caption in captions:
        word_caption = [dec_map[x]
                        for x in caption[:None if ed not in caption else
                            caption.index(ed)]]
        caption_sentences.append(' '.join(word_caption))
    return caption_sentences[0]

```


### Train for more epochs
In our observation, the training loss continues to decrease after our last submission on kaggle (470 epochs), and the public scores of the submissions are still decreasing, too.

This could imply that our model **has not converged**, so we believe that we can reach higher score if we starts working on this contest earlier.

----

# Hyper parameters setting
Here is the hyper parameters we use.


```python
def get_hparams():
    hparams = tf.contrib.training.HParams(
        vocab_size=vocab_size,
        batch_size=64,
        rnn_units=256,
        image_embedding_size=256,
        word_embedding_size=256,
        drop_keep_prob=0.7,
        lr=1e-3,
        training_epochs=50,
        max_caption_len=15,
        ckpt_dir='model_ckpt/sample')
    return hparams
```

----

# What we have attempted

## Pretrained word embedding
We tried to use the pre-trained word embedding matrix from spaCy. However, we were confronted with some problems when replacing the embedding matrix in the model with the one from spaCy because the dimension is different (word vector size).

Thus, we applied PCA to word vectors and transform it to the same size as that of original embedding matrix. We trained this model 200 epochs too (to compare with the original implementation and check how its performance is). It has a close (but a bit lower) score compared to the sample model.

## Attention
After trying to know how Attention works, we decided not to carry it out because we think that it's neccesary to implement the encoder part (CNN) instead of using the dataset preprocessed by TA, which could consume too much time to train.

----

# Demo
Here are some image captions of test data recorded during our training stage, for example:

- **300_epoch.csv** recorded the captions of our model when it was trained up to 300 epochs.
- **450_epoch_beam.csv** recorded that of our model when it reached 450 epochs and inference with beam search with beam size = 3.


```python
import pandas as pd
import os
from IPython.display import display, Image
import matplotlib.pyplot as plt

test_caption_files = [file for file in os.listdir('./test_demo/csv/') if file[-4:] == '.csv']
test_caption_files = sorted(test_caption_files)
print('The versions of captions recorded: ')
display(test_caption_files)

test_captions = pd.DataFrame.from_csv(os.path.join('./test_demo/csv/', test_caption_files[0]), index_col=None, header=0)
image_files = test_captions['img_id'][:6].values

for i, image_file in enumerate(image_files):
    print('test image:', image_file)
    display(Image(filename=os.path.join('./test_demo/pic/', image_file)))
    for file in test_caption_files:
        test_captions = pd.DataFrame.from_csv(os.path.join('./test_demo/csv/', file), index_col=None, header=0)
        print(file, ':', test_captions.iloc[i].values[1])
    print('\n')
```

    The versions of captions recorded: 



    ['200_epoch.csv',
     '250_epoch.csv',
     '300_epoch.csv',
     '350_epoch.csv',
     '450_epoch.csv',
     '450_epoch_beam.csv',
     '470_epoch_beam.csv']


    test image: 134297.jpg



<img src='pic/output_11_3.jpeg'>


    200_epoch.csv : a man holding a tennis racket on a tennis court
    250_epoch.csv : a woman walking down a street holding an umbrella
    300_epoch.csv : a man holding a umbrella while standing on a sidewalk
    350_epoch.csv : a woman walking down a street holding an umbrella
    450_epoch.csv : a man standing next to a woman holding a pink umbrella
    450_epoch_beam.csv : a woman walking down a street holding a pink umbrella
    470_epoch_beam.csv : a woman walking down a street holding an umbrella
    
    
    test image: 336695.jpg



<img src='pic/output_11_5.jpeg'>


    200_epoch.csv : a motorcycle parked on the side of a road
    250_epoch.csv : a motorcycle parked on the side of a road
    300_epoch.csv : a man riding a motorcycle down a street
    350_epoch.csv : a motorcycle parked on the side of a road
    450_epoch.csv : a man riding a motorcycle down a road
    450_epoch_beam.csv : a man riding a motorcycle down a road
    470_epoch_beam.csv : a man riding a motorcycle down a dirt road
    
    
    test image: 162144.jpg



<img src='pic/output_11_7.jpeg'>


    200_epoch.csv : a man in a red jacket skiing down a hill
    250_epoch.csv : a man and a woman standing on a ski slope
    300_epoch.csv : a man in a red jacket and a red jacket skiing
    350_epoch.csv : a man standing on a snow covered ski slope
    450_epoch.csv : a man and a woman standing on a snow covered slope
    450_epoch_beam.csv : a man standing on top of a snow covered ski slope
    470_epoch_beam.csv : a man standing on top of a snow covered ski slope
    
    
    test image: 321980.jpg



<img src='pic/output_11_9.jpeg'>


    200_epoch.csv : a cat is sitting on a laptop computer
    250_epoch.csv : a cat is sitting on a laptop computer
    300_epoch.csv : a cat sitting on a desk next to a laptop
    350_epoch.csv : a cat is sitting on a laptop computer
    450_epoch.csv : a cat sitting on a laptop computer on a desk
    450_epoch_beam.csv : a black cat laying on top of a laptop computer
    470_epoch_beam.csv : a cat sitting on top of a laptop computer on a desk
    
    
    test image: 397675.jpg



<img src='pic/output_11_11.jpeg'>


    200_epoch.csv : a herd of zebra standing on top of a lush green field
    250_epoch.csv : a herd of zebras standing in a field
    300_epoch.csv : a zebra standing in a field with a zebra in the background
    350_epoch.csv : a zebra standing in a field with a <RARE> in the background
    450_epoch.csv : a herd of zebra standing on top of a grass covered field
    450_epoch_beam.csv : a herd of zebra standing on top of a grass covered field
    470_epoch_beam.csv : a herd of zebra standing on top of a grass covered field
    
    
    test image: 284706.jpg



<img src='pic/output_11_13.jpeg'>


    200_epoch.csv : a toilet with a white seat on the floor
    250_epoch.csv : a toilet with a <RARE> seat on it
    300_epoch.csv : a toilet with a <RARE> seat on the floor
    350_epoch.csv : a bathroom with a toilet and a sink
    450_epoch.csv : a toilet with a <RARE> seat on the floor
    450_epoch_beam.csv : a white toilet sitting in a bathroom next to a sink
    470_epoch_beam.csv : a white toilet sitting in a bathroom next to a toilet
    
    


# Conclusion
In this contest, we apply many techonologies mentioned in class & notebook to see how it actually works.

Sometimes it doesn't work well as we expected. For example, we use **pre-trained word embedding** inside our model and fine tune it during the training process. We think it may perform better than a random initialized one (sample code in notebook), but its performance is not really far better than the sample one.

Besides, **beam search** works really well to improve the performance. We figure out how it works during this contest and are surprised at its simplicity and effectiveness.

Also, we learned many techniques from listening to other winners' sharing, for example, features extracted from different CNN architecture, fine tune CNN during training, change vocabulary size, etc, can give a better result for this task.

In the future contests, we should give those techniques a try and see how they work in our models.
