# ROS_Application

# Inhaltsverzeichnis
* [Overview](#task)
  * [Implementation](#implementation) 
  * [Documentation](#documentation) 
* [Task 1](#task-1)
  * [Objective 1](#objective-1)
  * [Proceeding 1](#proceeding-1)
* [Task 2](#task-2)
  * [Objective 2](#objective-2)
  * [Proceeding 2](#proceeding-2)
* [Task 3](#task-3)
  * [Objective 3](#objective-3)
  * [Proceeding 3](#proceeding-3)
* [Results](#results)

## Kurzübersicht
Ziel dieses Projektes war die Implementierung eines neuronalen Netzwerkes innerhalb einer ROS Umgebung, das in der Lage sein soll, eine Vorhersage darüber zu machen, welchen Wert eine handgeschriebene Ziffern von [0, 9] repräsentieren soll. Die verwendeten Bilddatenliegen der Einfachheit halber im PNG-Format vor, die Implementierung ließe jedoch mit geringfügigen Änderungen auch das Verarbeiten von VideoStreams zu diesem Zwecke zu. Das Programm wurde in Python 3 geschrieben.

- basic ROS Theorie
- Nodes, Topics, Messages
- Konzept von Neuronalen Netzen



### Documentation
Create a README.md with
* Inhaltsverzeichnis
* Bescheribung der Programame und ihrer Aufgaben
* Benutzungsanweisung und Benötigte Plugins
* Kurze Beschreibung des theoretischen Hintergrunds und andere Basics die notwendig sind das programm zu verstehen
  * Nodes, topics, messages, etc.
  * Concept of neural networks
  * etc.
* Beschreibung der Implementierung
* BEschreibung des Experiments und des Resultats/resultate -> siehe Vergleich merherer Architekturern
* Nutze Plots für bestimmte Aspekte Besonders den ROS graphen für das fertige Programm

## Task 1
### Objective 1
Implement a camera node which publishes a hardcoded image to a processing node, which is capable of image manipulation (resize, rgb to greyscale conversion, etc.). With each image, the camera node publishes simultaneously a hardcoded integer value to a new topic (of your choice). For image processing use suitable libraries. The processing node shall only subscribe to the image and not the integer topic. Further, the processed image shall be publish to a new topic (of your choice).

![Sketch - Graph](material/sketch_basic.png "Sketch - Graph")

### Proceeding 1
1. Read the beginner level tutorials 1.1, 1 to 10 for the basic concepts and features of ROS: [https://wiki.ros.org/ROS/Tutorials](https://wiki.ros.org/ROS/Tutorials)
2. Implement the talker-listener (a) and get it running (b).
   
   a. Tutorial 12: [https://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29](https://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29)  
 
   b. Tutorial 13: [https://wiki.ros.org/ROS/Tutorials/ExaminingPublisherSubscriber](https://wiki.ros.org/ROS/Tutorials/ExaminingPublisherSubscriber)  
 
   c. Instead of using rosrun, rewrite your application to make use of roslaunch and be capable of being started single-lined: [https://wiki.ros.org/roslaunch/XML](https://wiki.ros.org/roslaunch/XML)  
 
   d. **(optional)** examine your setup/application by some commands of tutorial 1-10
 
3. Change the talker node to send a hardcoded image (a) and and a random integer value (b) simultaneously to two different topics. You can use any image for this purpose. But don't push the resolution too far.  
    
   a. Instead of std_msgs.String use sensor_msgs.Image. Organize your package.xml ([https://wiki.ros.org/catkin/CMakeLists.txt](https://wiki.ros.org/catkin/CMakeLists.txt)) and CMakeList.txt ([https://wiki.ros.org/catkin/package.xml](https://wiki.ros.org/catkin/package.xml)) accordingly to the new imports.  
    
   b. One could use std_msgs.Int32 for defining the integer message but unfortunately primitive messages, such as float, int, etc. have no headers, which are required for later synchronization. So we need to define a custom message consisting of a header and an int32 variable (you only need the message part for now, services will be part of task 2):  
      [https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Introduction_to_msg_and_srv](https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Introduction_to_msg_and_srv)  
      
      **Note:**  
      Your *IntWithHeader.msg* file (or whatever you name it) will probably look like this:
```
Header header
int32 data
```
4. Change the listener node to subscribe on the image topic
 
5. Change the listener from just logging the received image to processing it. Crop it for example with [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html) 
 
6. Publish the processed image to a new topic of your choice (place this feature inside your processor node's subscribe-callback where you receive the image form the camera node)
 
7. Generate a picture of the ROS graph by RQT: [https://wiki.ros.org/rqt/Plugins](https://wiki.ros.org/rqt/Plugins)
 
8. Start documenting and describe the basic principles of ROS (theory).

## Task 2
### Objective 2
Use the setup of [task 1](#task-1) and implement a controller node, which subscribes to the processed image topic of the processor and the integer topic of the camera node. Synchronize the according inputs in the controller and save them as tupels in a data structure of your choice (e.g. list, dictionary, etc.). For the synchronization use ROS build-in methods. Do **not** build your own " dirty hacked" version of synchronization. Publish the image from the controller to a newly created AI service. The AI service receives the image from the controller node and responds with an integer value. For now, use a hardcoded value. This will be replaced by the prediction of the neural network later (also an integer value).

![Sketch - Graph](material/sketch_graph.png "Sketch - Graph")

### Proceeding 2
1. Create the controller node and use the ROS build-in TimeSynchronization functionality to receive the processed image from the processor node and the integer value from the camera node.  
 
   a. **(optional)** handle topics in general: [https://wiki.ros.org/message_filters](https://wiki.ros.org/message_filters)  
 
   b. TimeSynchronization: [https://wiki.ros.org/message_filters#Example_.28Python.29-1](https://wiki.ros.org/message_filters#Example_.28Python.29-1) (4.3)  
      
      **Note:**  
      When using synchronization, a message with a header is required. The primitive messages such as int32 do not have headers. Due to this reason, we defined our custom message in task 1.
 
2. Save the inputs by a data structure of your choice in your controller node (e.g. list, dict, etc.).
 
3. Create a ROS service for the AI prediction task. Therefore  
 
   a. Include the service tutorial (two parts) in your setup and run it to be sure your basic service works.  
      **Service Tutorials**  
      Part 1: [https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_srv](https://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_srv)  
      Part 2: [https://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29](https://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29)        
      
      **Note:**  
      Some classic errors are: no execution permission set on file, forgot to rebuild caktin workspace or simply not sourcing newly build devel directory after catkin\_make. In rare setups caching issues occure. To avoid them, manually delete build and devel directories before executing catkin\_make.  
   
   b. Adapt the service to receive an image and return an integer value.
      
      **Note:**   
      This is by far more easy than it seems. Again, do not make a dirty hack. Just wrap the ROS sensor\_msgs/Image message in your service message. See the service tutorial part 1 with an example of a msg that uses a Header, a string primitive, and two other msgs. You simply need the sensor_msg/image. Adapt the rest of the service accordingly. For the moment, use a hardcoded integer as return value of the service. This will be replaced by the prediction of the neuronal network later.  
      Your *AI.srv* file (or whatever you name it) will probably look like this:
```
sensor_msgs/Image image
---
int32 result
```      

and your imports in python like this:
      
```python
from beginner_tutorials.srv import AI, AIResponse
```
      
4. Send the image from the controller to the AI service and save the returned integer into a temporary variable.  
 
5. Run your application and generate a picture of the ROS graph by RQT: [https://wiki.ros.org/rqt/Plugins](https://wiki.ros.org/rqt/Plugins)
 
6. Continue documenting and describe the basic principles of ROS (theory). Further, describe the concept and implementation of your current application (the graph setup is complete at this state. Only changes inside functions will happen from now on, regarding the AI and according image processing methods).

## Task 3
### Objective 3
Use a framework of your choice that supports Python3 (ROS dependency) to train a model on the MNIST dataset and save it. Afterwards, adapt the processing node to identically transform the images based on your training transformation. Adapt your ROS application to load your model and predict the digit received in the AI service. Automatically compare the received prediction with the stored true value in the controller node.  
As last step, make your input random in the camera node by generating a number between 0-9 and load the image accordingly. Finally, check if your set up is still synchronized and that everything works as expected.

### Proceeding 3
The proceeding is split into three parts. The training, the prediction and evaluation. At first you will train a model, which is saved and loaded later on by the ROS application to predict the digit of a given image.

#### Train and Save Model 
In order to predict the digit on the received image from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), a model needs to be trained. Therefor you can use any library you like to, as long as it is compatible to Python3 (due to ROS). Large frameworks like [PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) or [Keras/Tensorflow](https://www.tensorflow.org/tutorials/keras/classification) and similiar support Python3. Also those frameworks offer MNIST dataset loaders for training.  

1. Create a fully connected network by the framework of your choice
2. Set up or use predefined processing for the MNIST dataset
3. Train your model on the MNIST dataset
4. Save the model
5. Document your processing and training pipeline (implementation) and the training process (theory)

**Note:**  
As an example, you can have a look at [this implementation](https://gitlab.com/baumannpa_teaching/pytorch-mnist). 

#### Load Model and Predict
1. Replace your custom image by loading a [MNIST image](material/mnist_images) in your camera node (use the camera node's static integer value as image name).
2. Process the image in the processing node the same way your training pipeline did (pay attention that you understand the transformation process, this is an important part for documentation). Since ROS image messages are integer based, you can handle the normalization inside the AI service (0 to 1 values as floats).
3. Load your model in the AI service and replace the hardcoded value by the prediction of the model.  
**Note:**  
The model has to be loaded, not being trained on each program start. MNIST is a simple computation task but real life tasks can take up to months of training for a model. Further, the service shall return an integer value not an one-hot encoded vector.
4. Make an automatized comparison between the predicted and true value in the controller node
5. In the camera node, replace the static input image by a random one. Therefor use a random generated number and load an according [MNIST image](material/mnist_images).
6. Check if your ROS application is working as expected.
7. Document your implementation.

#### Evaluating Different Setups
1. Train a few models (atleast two) with different setups (e.g. layer, optimizer, activation function, hyperparameter, etc.)
2. Compare them and describe your results (use plots/diagrams if they clarify certain aspects) 
 
## Results
Submit your
* ROS application and README.md
* scripts, notebooks, etc. whatever you used to train your models

to moodle or provide a public git link. Do not include your trained models. Only code and readme. Your models' results have to be reproducable based on your provided code and documentation. 
