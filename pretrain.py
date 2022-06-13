import os
import pickle
import torch.nn as nn
import torch
import time
import argparse
import nltk
import torch.optim as opt
import json

from torch.nn.utils.rnn import pack_padded_sequence
from model_vgg19 import EncoderCNN, DecoderLSTM
from torchvision import transforms
from dataloader import get_loader
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # with open(args.vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)

    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    encoder = EncoderCNN().to(device)

    for i, param in enumerate(encoder.parameters()):
        param.requires_grad = False

    decoder = DecoderLSTM(decoder_dim=args.decoder_dim, att_dim=args.att_dim, vocab_size=len(word_map),
                          embed_dim=args.embed_dim, dropout=args.dropout).to(device)

    encoder_optimizer = None  # 不fine-tune VGG
    decoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr)

    decoder_lr = args.decoder_lr

    criterion = nn.CrossEntropyLoss().to(device)

    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_loader = get_loader(data_folder=args.data_folder, data_name=args.data_name, split='TRAIN',
                              transform=transform,
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_loader(data_folder=args.data_folder, data_name=args.data_name, split='VAL', transform=transform,
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    best_bleu4 = 0

    print('start training...')
    for epoch in range(args.epochs):
        if epoch == 5:
            decoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr / 2)
        if epoch == 8:
            decoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr / 4)
        train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, args.print_step)
        bleu_4 = val(val_loader, encoder, decoder, criterion, args.print_step, word_map)
        is_best = bleu_4 > best_bleu4
        best_bleu4 = max(best_bleu4, bleu_4)

        save_checkpoint(args.save_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu_4, is_best)


def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, print_step):
    encoder.train()
    decoder.train()
    start_time = time.time()
    for i, (images, captions, lens) in enumerate(train_loader):
        images = images.to(device)          # [64, 3, 224, 224]
        captions = captions.to(device)      # [64, max_len_t]
        lens = lens.to(device)              # [64, 1]

        images = encoder(images)            # [64, 14, 14, 512]
        predictions, alphas, caps_sorted, decode_lens, sort_ind = decoder(images, captions, lens)

        targets = caps_sorted[:, 1:]        # pred: [batch_size, max_len-1, vocab_size], tar: [batch_size, max_len_t-1]

        # predictions, _ = pack_padded_sequence(predictions, decode_lens, batch_first=True)
        # targets, _ = pack_padded_sequence(targets, decode_lens, batch_first=True)

        predictions = pack_padded_sequence(predictions, decode_lens, batch_first=True)[0]

        targets = pack_padded_sequence(targets, decode_lens, batch_first=True)[0]   # pytorch1.1.0后只能这样，不然报错

        loss = criterion(predictions, targets)                                      # [b_new, vocab_size], [b_new]
        loss += args.lbd * ((1. - alphas.sum(dim=1)) ** 2).mean()                   # 双重随机注意力正则化 对时间步维度求和 靠近1

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        if i % print_step == 0:
            end_time = time.time()
            print('Epoch: {}, step: {}/{}, loss: {}, time: {}'.format(epoch, i, len(train_loader), loss,
                                                                      end_time - start_time))
            start_time = time.time()


def val(val_loader, encoder, decoder, criterion, print_step, word_map):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    references = list()
    hypotheses = list()

    start_time = time.time()
    with torch.no_grad():
        for i, (images, captions, lens, all_caps) in enumerate(val_loader):
            images = images.to(device)          # [b, 3, 224, 224]
            captions = captions.to(device)      # [b, max_len_t]
            lens = lens.to(device)              # [b, 1]
            all_caps = all_caps.to(device)      # [b, 5, max_len_t]

            images = encoder(images)
            predictions, alphas, caps_sorted, decode_lens, sort_ind = decoder(images, captions, lens)

            targets = caps_sorted[:, 1:]

            predictions_copy = predictions.clone()
            predictions = pack_padded_sequence(predictions, decode_lens, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lens, batch_first=True)[0]

            loss = criterion(predictions, targets)
            loss += args.lbd * ((1. - alphas.sum(dim=1)) ** 2).mean()  # 双重随机注意力 正则化

            if i % print_step == 0:
                end_time = time.time()
                print('step: {}/{}, loss: {}, time: {}'.format(i, len(val_loader), loss, end_time - start_time))
                start_time = time.time()

            # References
            all_caps = all_caps[sort_ind]                       # because images were sorted in the decoder
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()                 # [5, max_len_t]
                img_captions = list(                            # map函数 不改变原列表 返回新的列表
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))                              # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(predictions_copy, dim=2)       # [b, max_len-1]
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lens[j]])    # remove pads
            preds = temp_preds
            hypotheses.extend(preds)                            # append直接加在最后 extend则是拆掉一层再加在最后

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        print('BLEU4: {}'.format(bleu4))

    return bleu4


def save_checkpoint(save_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    state = {'epoch': epoch,
             'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': None,
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = 'checkpoint_' + 'epoch_{}'.format(epoch) + '.pth'
    save_path = os.path.join(save_path, filename)
    torch.save(state, save_path)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, save_path + '_BEST_BLEU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder_dim', type=int, default=768, help='hidden size of LSTM')
    parser.add_argument('--att_dim', type=int, default=512, help='dim of att')
    parser.add_argument('--embed_dim', type=int, default=256, help='dim of word embeddings')

    parser.add_argument('--data_folder', type=str,
                        default='autodl-tmp/coco/images', help='dir of coco images')
    parser.add_argument('--data_name', type=str,
                        default='coco_5_cap_per_img_5_min_word_freq', help='data name of json and hdf5')

    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--decoder_lr', type=float, default=5e-4, help='lr of decoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--lbd', type=float, default=0.2, help='lambda')
    parser.add_argument('--save_path', type=str, default='autodl-tmp/models', help='path for saving models')
    parser.add_argument('--print_step', type=int, default=100, help='time step for printing states')

    args = parser.parse_args()
    main(args)

