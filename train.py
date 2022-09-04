import torch
import torchvision
import torchvision.transforms as transforms
from Dataset.dataset import OliveCropsDataset
from Models.convolutional_models import Net
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Transformation pipeline
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5,
                                                     0.5)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
dataset_name = 'Data50x50'
dataset_csv_path = 'Dataset/'+dataset_name+'/DatasetCrop.csv'
dataset_images_path = 'Dataset/'+dataset_name+'/images'

trainset = OliveCropsDataset(dataset_csv_path, root_dir=dataset_images_path, learn_set='training', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = OliveCropsDataset(dataset_csv_path, root_dir=dataset_images_path, learn_set='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('superintensivo', 'intensivo', 'tradicional')


# functions to show an image
def imshow(img):
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()  # Transform to numpy array
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# function to evaluate the model #
def evaluate_model(dataloader, model, loss_fn):

	with torch.no_grad():
		model.eval()
		# Sample the dataloader #
		new_data, true_labels = iter(dataloader).next()
		# Predict #
		predicted_values = model(new_data.to(device)).cpu()
		evaluation_loss = loss_fn(predicted_values, true_labels)
		_, predicted_classes = torch.max(predicted_values, 1)

		accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_classes)

	# Return to train mode #
	model.train()

	return evaluation_loss, accuracy


example_image, _ = trainset[0]

# Create the network
net = Net(input_size=tuple(example_image.shape), output_size=len(classes))

print(net)

# Create the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4,)

number_of_minibatches = len(trainset) / batch_size
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=50, steps_per_epoch=len(trainloader))

pbar = tqdm(range(50))
running_loss = 0.0
test_data = (0.0, 0.0)

lrs = [0.01]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

for epoch in pbar:  # loop over the dataset multiple times

	net.train()

	pbar.set_description(f"Epoch {epoch} | Avg. Loss: {running_loss/len(trainloader)} | Test loss: {test_data[0]} | Accuracy: {test_data[1]} | LR: {lrs[-1]}")

	running_loss = 0.0

	for i, data in enumerate(trainloader):

		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs.to(device))
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# Update the lr
		sched.step()

		lrs.append(get_lr(optimizer))

		running_loss += loss.item()

	""" Evaluate the model with the test set """
	test_data = evaluate_model(dataloader=testloader, model=net, loss_fn=criterion)


print('Finished Training')

# Save the model #
PATH = './Models/trained_models/classifier'+dataset_name+'.pth'
torch.save(net.state_dict(), PATH)


plt.plot(lrs)
plt.savefig('./learning_rate')