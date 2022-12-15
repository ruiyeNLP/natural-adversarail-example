import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel
import numpy as np


def predict_for_file(input_file, output_file, model, batch_size=32, to_normalize=False):
    test_data = read_lines(input_file)
    predictions = []
    cnt_corrections = 0
    batch = []
    correction_labels = []
    arr = []
    num = []
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, edits, cnt = model.handle_batch(batch)
            correction_labels.extend(edits)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, edits, cnt = model.handle_batch(batch)
        correction_labels.extend(edits)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    cnt_corr_sentences = len(correction_labels)
    test = np.array(correction_labels)
    print(test.shape)
    print(np.array(predictions).shape)
    all_corr_labels = [i for j in correction_labels for i in j]
    data = np.array(all_corr_labels)
    arr, num = np.unique(data, return_counts=True)


    with open(output_file, 'w') as f:
        f.write("\n".join(result_lines) + '\n')
    return cnt_corrections, arr, num, cnt_corr_sentences

def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    # cnt_corrections = predict_for_sentences(sentences, model)
    cnt_corrections,arr,num, cnt_corr_sentences = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size, 
                                       to_normalize=args.normalize)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")
    print(f"Produced overall sentences: {cnt_corr_sentences}")
    print(arr)
    print(num)

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()
    main(args)
# python test_sentence.py --model_path ./model/roberta_1_gectorv2.th --vocab_path ./data/output_vocabulary --input_file ./mr/processed.txt --output_file ./mr/processed_corrected.txt


# def predict_for_sentences(sentences, model, batch_size=32, to_normalize=False):
#     # test_data = read_lines(input_file)
#     predictions = []
#     cnt_corrections = 0
#     batch = []
#     correction_labels = []
#     for sent in sentences:
#         batch.append(sent.split())
#         if len(batch) == batch_size:
#             preds, edits, cnt = model.handle_batch(batch)
#             predictions.extend(preds)
#             correction_labels.extend(edits)
#             cnt_corrections += cnt
#             batch = []
            
#     if batch:
#         preds, edits, cnt = model.handle_batch(batch)
#         predictions.extend(preds)
#         correction_labels.extend(edits)
#         cnt_corrections += cnt


#     result_lines = [" ".join(x) for x in predictions]
#     if to_normalize:
#         result_lines = [normalize(line) for line in result_lines]

#     # for sentence, label in zip(predictions, correction_labels):
#     #     print('***************')
#     #     print(correction_labels)
#     #     print(sentence)
#     #     print('***************')
#     # cnt_corr_sentences = len(correction_labels)
#     cnt_corr_sentences = len(correction_labels)
#     test = np.array(correction_labels)
#     print(correction_labels)
#     print(test.shape)
#     print(np.array(predictions).shape)
#     all_corr_labels = [i for j in correction_labels for i in j]
#     data = np.array(all_corr_labels)
#     arr, num = np.unique(data, return_counts=True)
#     print(cnt_corr_sentences)
#     # print(all_corr_labels)
#     print(arr)
#     print(num)
#     # print(correction_labels)
#     # print(result_lines)
#     # with open(output_file, 'w') as f:
#     #     f.write("\n".join(result_lines) + '\n')
#     return cnt_corrections

# def main(args):
#     # get all paths
#     model = GecBERTModel(vocab_path=args.vocab_path,
#                          model_paths=args.model_path,
#                          max_len=args.max_len, min_len=args.min_len,
#                          iterations=args.iteration_count,
#                          min_error_probability=args.min_error_probability,
#                          lowercase_tokens=args.lowercase_tokens,
#                          model_name=args.transformer_model,
#                          special_tokens_fix=args.special_tokens_fix,
#                          log=False,
#                          confidence=args.additional_confidence,
#                          del_confidence=args.additional_del_confidence,
#                          is_ensemble=args.is_ensemble,
#                          weigths=args.weights)

#     sentences = [
#         'The company intends to provide further updates on this pending merger shortly as it continues with its due diligence process congratulations',
#         'The fact that many have lost their jobs , their homes , their dreams in these difficult times confirms for us that life carries with it a Good Friday experience -- that darkness and disappointment can be constant companions .',
#         'This article was first published on guardian.co.uk at 12.07 BST days Saturday 11 April 2009 .',
#         'Pole-position qualifying its that saturday',
#         'He is accused of running his office like frat house , where cursing and harassing young female staffers was the norm .',
#         'Given this new reputation for openness . his endorsement of Gordon Brown is all the more significant as voters are unlikely to think that John Prescott is the sort of politician who says one thing public and quite another in private .'
#     ]
#     cor_sentences = [
#         'The company intends to provide further updates on this pending merger shortly as it continues with its due diligence process congratulations',
#         'The company intends to provide further updates on this pending merger shortly as it continues with its due diligence process congratulations', 
#         'The fact that many have lost their jobs , their homes , their dreams in these difficult times confirms for us that life carries with it a Good Friday experience -- that darkness and disappointment can be constant companions .',
#         'The fact that many have lost their jobs , their homes , their dreams in these difficult situations confirms for us that life carries with it a Good Friday experience and that darkness and disappointment can be constant companions .', 
#         'This article was first published on guardian.co.uk at 12.07 BST days Saturday 11 April 2009 .',
#         'This article was first published on guardian.co.uk at 12.07 BST on Saturday 11 April 2009 .', 
#         'Pole-position qualifying its that saturday',
#         'Pole-position qualifying that Saturday', 
#         'He is accused of running his office like frat house , where cursing and harassing young female staffers was the norm .',
#         'He is accused of running his office like a house , where cursing and harassing young female staffers was the norm .', 
#         'Given this new reputation for openness , his endorsement of Gordon Brown is all the more significant as voters are unlikely to think that John Prescott is the sort of politician who says one thing in public and quite another in private .'
#         'Given this new reputation for openness . his endorsement of Gordon Brown is all the more significant as voters are unlikely to think that John Prescott is the sort of politician who says one thing public and quite another in private .'
#     ]


#     cnt_corrections = predict_for_sentences(sentences, model)
#     # cnt_corrections,arr,num = predict_for_file(sentences, model)
#     # evaluate with m2 or ERRANT
#     print(f"Produced overall corrections: {cnt_corrections}")



# if __name__ == '__main__':
#     # read parameters
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path',
#                         help='Path to the model file.', nargs='+',
#                         required=True)
#     parser.add_argument('--vocab_path',
#                         help='Path to the model file.',
#                         default='data/output_vocabulary'  # to use pretrained models
#                         )
#     # parser.add_argument('--input_file',
#     #                     help='Path to the evalset file',
#     #                     required=True)
#     # parser.add_argument('--output_file',
#     #                     help='Path to the output file',
#     #                     required=True)
#     parser.add_argument('--max_len',
#                         type=int,
#                         help='The max sentence length'
#                              '(all longer will be truncated)',
#                         default=50)
#     parser.add_argument('--min_len',
#                         type=int,
#                         help='The minimum sentence length'
#                              '(all longer will be returned w/o changes)',
#                         default=3)
#     parser.add_argument('--batch_size',
#                         type=int,
#                         help='The size of hidden unit cell.',
#                         default=128)
#     parser.add_argument('--lowercase_tokens',
#                         type=int,
#                         help='Whether to lowercase tokens.',
#                         default=0)
#     parser.add_argument('--transformer_model',
#                         choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
#                                  'bert-large', 'roberta-large', 'xlnet-large'],
#                         help='Name of the transformer model.',
#                         default='roberta')
#     parser.add_argument('--iteration_count',
#                         type=int,
#                         help='The number of iterations of the model.',
#                         default=5)
#     parser.add_argument('--additional_confidence',
#                         type=float,
#                         help='How many probability to add to $KEEP token.',
#                         default=0)
#     parser.add_argument('--additional_del_confidence',
#                         type=float,
#                         help='How many probability to add to $DELETE token.',
#                         default=0)
#     parser.add_argument('--min_error_probability',
#                         type=float,
#                         help='Minimum probability for each action to apply. '
#                              'Also, minimum error probability, as described in the paper.',
#                         default=0.0)
#     parser.add_argument('--special_tokens_fix',
#                         type=int,
#                         help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
#                              'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
#                         default=1)
#     parser.add_argument('--is_ensemble',
#                         type=int,
#                         help='Whether to do ensembling.',
#                         default=0)
#     parser.add_argument('--weights',
#                         help='Used to calculate weighted average', nargs='+',
#                         default=None)
#     parser.add_argument('--normalize',
#                         help='Use for text simplification.',
#                         action='store_true')
#     args = parser.parse_args()
#     main(args)

# # python test_sentence.py --model_path ./model/roberta_1_gectorv2.th --vocab_path ./data/output_vocabulary