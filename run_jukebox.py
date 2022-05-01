import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()

print("finished imports")

# Sample from the 5B or 1B Lyrics Model

model = "1b_lyrics" # or "5b_lyrics"
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model=='5b_lyrics' else 3
hps.name = 'samples'
chunk_size = 16 if model=="5b_lyrics" else 32
max_batch_size = 3 if model=="5b_lyrics" else 16
hps.levels = 3
hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

print("created model")

# Specify your choice of artist, genre, lyrics, and length of musical sample.

sample_length_in_seconds = 90          # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                       # range work well, with generation time proportional to sample length.  
                                       # This total length affects how quickly the model 
                                       # progresses through lyrics (model also generates differently
                                       # depending on if it thinks it's in the beginning, middle, or end of sample)

hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

metas = [dict(artist = "ed sheeran",
            genre = "Pop",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """She said she broke it down
And then collected them like model ships
We danced through the rain
Her jaw was set and her eyes were blue


And now she's leaving on an eastbound train
To find her youth in the west
But neither one of us will know where
She sleeps or who she kisses there
It's hard to leave an old love alone
Just hard to leave her

And now she's over the earth
On a westbound train
In a place where the years have no number
She sits alone, she sits alone
She sits alone, she sits alone


From birth until forever
No words are ever lost
The future frozen in the present
The pens that I hold
Take my future from me
But you can touch my heart
And support it alone

And now she's leaving on an eastbound train
To find her youth in the west
But neither one of us will know where
She sleeps or who she kisses there
It's hard to leave an old love alone
Just hard to leave her
And now she's over the waves
And sleeping in them too
She drives this shell
And goes on a westbound train
            """,
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

sampling_temperature = .98

lower_batch_size = 16
max_batch_size = 3 if model == "5b_lyrics" else 16
lower_level_chunk_size = 32
chunk_size = 16 if model == "5b_lyrics" else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True,
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

print("set parameters")
print("starting level 2 sampling")

zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
# t.save(torch.FloatTensor(zs), "level2.pt")
#data = t.load("samples/level_2/data.pth.tar", map_location='cpu')
#zs = [z.cuda() for z in data['zs']]
#assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
#del data

print("finished level 2")

# Set this False if you are on a local machine that has enough memory (this allows you to do the
# lyrics alignment visualization during the upsampling stage). For a hosted runtime,
# we'll need to go ahead and delete the top_prior if you are using the 5b_lyrics model.
if False:
  del top_prior
  empty_cache()
  top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

print("finished loading upsamplers")
print("starting upsampling")

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

print("finished upsampling")

del upsamplers
empty_cache()
