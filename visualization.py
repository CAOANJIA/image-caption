import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from model_vgg19 import *
# from scipy.misc import imread, imresize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))

    encoder = EncoderCNN().to(device)

    decoder = DecoderLSTM(decoder_dim=768, att_dim=512, vocab_size=len(word_map),
                          embed_dim=256, dropout=0.5).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, smooth=True)


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    img = Image.open(image_path)
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width

    img = img.crop((left, top, right, bottom))
    img = img.resize([224, 224], Image.ANTIALIAS)  # ANTIALIAS 是一个过滤器

    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.
    img = torch.FloatTensor(img).to(device)

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([normalize])

    image = transform(img)                              # [3, 224, 224]

    image = image.unsqueeze(0)                          # [1, 3, 224, 224]

    encoder_out = encoder(image)                        # [1, 14, 14, encoder_dim]
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    encoder_out = encoder_out.view(1, -1, encoder_dim)                          # [1, num_pixels, encoder_dim]
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)                # [k, num_pixels, encoder_dim]

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)     # [k, 1]

    seqs = k_prev_words                                                         # [k, 1]

    top_k_scores = torch.zeros(k, 1).to(device)                                 # [k, 1]

    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)    # [k, 1, 14, 14] 初始化alpha为1

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_h_c(encoder_out)

    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)             # [s, embed_dim]

        att_out, alpha = decoder.attention(encoder_out, h)                  # [s, encoder_dim], [s, num_pixels]

        alpha = alpha.view(-1, enc_image_size, enc_image_size)              # [s, 14, 14]

        beta = decoder.sigmoid(decoder.f_beta(h))
        att_out = beta * att_out

        h, c = decoder.decode_step(torch.cat([embeddings, att_out], dim=1), (h, c))     # [s, decoder_dim]

        scores = decoder.fc(h)                                                          # [s, vocab_size]
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores                                # [s, vocab_size]

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)                # [s]
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)          # [s]

        prev_word_inds = top_k_words / vocab_size                                       # [s]
        next_word_inds = top_k_words % vocab_size                                       # [s]

        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)     # [s, step+1]
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)                                                           # [s, step+1, 14, 14]

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        print(top_k_scores)
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(os.path.join(args.save_path, '166995.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i',
                        default='D:/datasets/coco/images/val2014/COCO_val2014_000000166995.jpg', help='path to image')
    parser.add_argument('--model', '-m',
                        default='D:/models/Show, Attend and Tell/checkpoint_finetune_epoch_4.pth', help='path to model')
    parser.add_argument('--word_map', '-wm',
                        default='D:/datasets/coco/images/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=3, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--save_path', default='D:/', help='img save path')
    args = parser.parse_args()
    main(args)
