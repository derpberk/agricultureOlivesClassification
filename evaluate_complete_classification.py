import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Dataset.dataset import OliveCompleteDataset
from Models.convolutional_models import Net
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from Dataset.image_utilities import sliding_window, pad_if_necessary
import numpy as np
from tqdm import tqdm

import matplotlib.patches as patches


crop_size = (120, 120)
colors=['r','g','b']

dataset_csv_path = 'Dataset/Original Data/OriginalImagesDataset.csv'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
validationset = OliveCompleteDataset(dataset_csv_path, root_dir='Dataset', learn_set='evaluation', transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=True, num_workers=0)
classes = ('superintensivo', 'intensivo', 'tradicional')

def evaluate_one_image(img, mask, window_size, stride, prediction_model, true_val):

	class_score = [0, 0, 0]

	#plt.imshow(img[0], cmap='gray')

	for x, y, crop in sliding_window(img[0], stepSize=stride, windowSize=window_size):

		padded_mask = pad_if_necessary(mask[y:y + window_size[1], x:x + window_size[0]], minimum_size=window_size)
		level_of_belonging = padded_mask.mean()/255.0
		if 0.1 < level_of_belonging:
			padded_crop = pad_if_necessary(crop, minimum_size=window_size)
			transformed = transform(padded_crop)
			prediction = prediction_model(transformed.unsqueeze(0).float()).cpu()
			_, predicted = torch.max(prediction, 1)
			class_score[predicted.item()] += 1
			#color = colors[predicted_class.item()]
			#rect = patches.Rectangle((x, y), window_size[0], window_size[1], linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
			# Add the patch to the Axes
			#plt.gca().add_patch(rect)

	predicted_class_overall = np.asarray(class_score).argmax()

	"""
	if true_val != predicted_class_overall:
		plt.imshow(img[0],cmap='gray')
		plt.show()
	
	
	"""

	#plt.show()
	return predicted_class_overall

model = Net(input_size=crop_size, output_size=len(classes))
model.load_state_dict(torch.load(f'./Models/trained_models/classifierData{crop_size[0]}x{crop_size[1]}.pth'))
model.eval()

with torch.no_grad():

	predicted_classes = []
	true_labels = []

	for indx in tqdm(range(len(validationset))):

		img, mask, true_label = validationset[indx]
		predicted_class = evaluate_one_image(img, mask, window_size=crop_size, stride=crop_size[0]//4, prediction_model=model, true_val=true_label)
		predicted_classes.append(predicted_class)
		true_labels.append(true_label)

	accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_classes)
	macro_precision = precision_score(y_true=true_labels, y_pred=predicted_classes, average='macro')
	f1 = f1_score(y_true=true_labels, y_pred=predicted_classes, average='macro')
	confusion = confusion_matrix(y_true=true_labels, y_pred=predicted_classes)

	print(f"Precision: {macro_precision}, Accuracy: {accuracy}")

	disp = ConfusionMatrixDisplay(confusion, display_labels=classes)

	disp.plot(cmap='Blues')

	plt.show()
