# CNN
Convolution Neural Network implemented by tensorflow

## Definition
Convolution, a rolled up or coiled condition이라는 뜻을 가졌다.
CNN은 이런 convolution 단어에 뜻에따라서, data feature를 vector형식이 아닌 matrix 형식으로 받아, 이를임의의 kernel size를 가진 filter를convolution하여 output으로 matrix 형식의 activation map(feature map)을 반환하는 Convolution layer를 갖고 마지막에 Fully Connected layer를 통해 feature가 one_hot encoding된다.

## Terminology
- Convolution layer
- Activation layer(ReLU)
Model에 non-linearity를 더하는 과정으로써 activation fuction으로 sigmoid,tanh,ReLU 등이 존재한다.이 예제에서는 ReLU를 사용하는데, 그 이유는 sigmoid는 back propagtion을 진행하며 chain rule을 적용해가며 점차 Vanishing Gradient 현상이 나타나 학습이 이루어지지 않는 문제가 있는 반면, ReLU는 미분의 계산량도 간편할 뿐만 아니라 기울기가 사라지는 문제도 해결할 수 있기 떄문이다.
- Pooling layer
Activation Map에서 subsampling하는 과정을 의미한다. subsampling function으로 평균값 or 최댓값을 취하는 방법이 있는데, 주로 Max_pooling을 사용한다. 그 이유는 class label에 가장 영향력이 큰 feature를 선택함으로써 Model이 좀 더 Robust해지고 overfitting을 방지하며 계산량까지 줄어드는 장점이 있다.
주로 kernel size : 2x2 , stride : 2를 사용한다.
- Filter
Convolution layer에서 stride하는 단위 matrix를 의미하며 Weight,Kernel과도 같은 의미로 쓰인다.
Filter는 대응되는 Input layer에 대한 feature의 특징을 나타내는 matrix이고 학습을 진행하게 될때마다 각 weight값이 변경된다. filter는 input layer의 channel(depth)와 같은 depth를 가져 다음과 같이 크기를 표기한다. F : filter matrix 가로,세로 길이 [F,F,depth,# of filter]
- Activation Map(Feature Map)
Input layer가 filter에 의해 mapping되어 나온 output layer를 의미한다. Activation Map의 depth는 filter의 갯수와 같고 output size는 다음고 같은 식으로 계산된다.
(inputsize + 2*paddingsize - filtersize / stride) + 1 = outputsize
- Fully Connected layer
multi-channel convolution layer를 vector로 합하는 과정을 의미한다.
- Stride
filter가 움직이는 단위를 의미한다.
- Padding
convolution layer를 거칠 때 output size가 점차 줄어들어 input에 비해 정보가 손실되는 문제가 있는데, 이를 input에 추가적인 padding을 적용해 outputsize와 inputsize를 갖게 하여 정보의 손실을 막는 것을 뜻한다.
- Receptive Field
- filter에 대응되는 input matrix를 의미한다.

## 주요 소스코드
<pre><code>
# layer1
W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01)) # filter(kernel)
L1 = tf.nn.conv2d(X_img,W1,strides=[1,1,1,1],padding='SAME') # convolution layer
L1 = tf.nn.relu(L1) # activation layer
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # max pooling layer
L1 = tf.nn.dropout(L1,keep_prob=0.7) # dropout
# 14*14*32
# layer2
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # 7*7*64
L2 = tf.nn.dropout(L2,keep_prob=0.7)

# ...생략 ...
# Fully Connected Layer
# layer4
L3 = tf.reshape(L3,shape=[-1,4*4*128])
W4 = tf.get_variable('W4',shape=[4*4*128,625],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.matmul(L3,W4)+b4
L4 = tf.nn.dropout(L4,keep_prob=0.5)
</code></pre>

# Ensemble
CNN 모델을 한개가 아닌 여러개를 만들어, 최종적으로 얻어진 hypothesis를 모두 취합하여 test-data에 대한 class label을 예측하는 방법으로 단일 모델을 사용하는 것보다 accuracy를 향상시킬 수 있다. 위 예제에서는 Ensemble방법으로 각 모델을 통해 얻어진 hypothesis를 sum하는 방법을 사용했다.

<pre><code>
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size*10).reshape((test_size,10))
for idx,m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images) # return logits as list
    predictions += p

prediction = tf.argmax(predictions,1)
is_correct = tf.equal(prediction,tf.argmax(mnist.test.labels,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,dtype=tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
</code></pre>

