from bertRelation.tasks.infer import infer_from_trained, FewRel
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel')
parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT',help="training data .txt file path")
parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT',help="test data .txt file path")
parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT 1 - ALBERT 2 - BioBERT''')
parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', 'bert-large-uncased',For ALBERT: 'albert-base-v2','albert-large-v2' For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")

args = parser.parse_args()                                                                                  
inferer = infer_from_trained(args, detect_entities=False)
test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
result=inferer.infer_sentence(test, detect_entities=False)
print(result)