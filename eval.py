from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import time

file = open('eval_log.txt', 'w+')
# file = open('compare_models_log.txt', 'w+')
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# checkpoint = './checkpoints/SSD_940_9809.pth'

checkpoint = './checkpoints/2020-05-29-13'
checkpoint2 = './checkpoints/SSD_880_6269.pth'
checkpoint3 = './checkpoints/SSD_900_7449.pth'
checkpoint4 = './checkpoints/SSD_930_9219.pth'
checkpoint5 = './checkpoints/SSD_940_9809.pth'
checkpoints = [checkpoint, checkpoint2, checkpoint3, checkpoint4, checkpoint5]

total_time = 0

def load(checkpoint):
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)

    # Switch to eval mode
    model.eval()

    # Load test data
    test_dataset = PascalVOCDataset(data_folder,
                                    split='test',
                                    keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
    return test_loader, model


def evaluate(test_loader, model, min_score, max_overlap, top_k):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    global total_time
    n = 0

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            time_start = time.time()
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=min_score, max_overlap=max_overlap,
                                                                                       top_k=top_k)
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            total_time += time_end - time_start
            n += 1
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    print('avg_time :', total_time / n)
    # Print AP for each class
    # print('det_boxes: ', det_boxes)
    # print('true_boxes: ', true_boxes)
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    max_overlap = 0.1
    score = 0.01
    top_k = 100

    # for checkpoint in checkpoints:
    #     test_loader, model = load(checkpoint)
    #     evaluate(test_loader, model, 0.01, 0.45)
    #     print('model: ', checkpoint, file=file)
    test_loader, model = load(checkpoint)
    evaluate(test_loader, model, 0.3, 0.45, 200)
    # while score < 1:
    #     print('Min score = ', score, file=file)
    #     evaluate(test_loader, model, score, 0.45, 200)
    #     score += 0.05
    # while max_overlap < 1:
    #     print('max_overlap = ', max_overlap, file=file)
    #     evaluate(test_loader, model, 0.3, max_overlap, 200)
    #     max_overlap += 0.05
    # while top_k <= 500:
    #     print('top_k = ', max_overlap, file=file)
    #     evaluate(test_loader, model, 0.3, 0.45, top_k)
    #     top_k += 50
