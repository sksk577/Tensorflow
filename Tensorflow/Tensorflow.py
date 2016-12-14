# mnist의 데이터를 코드 안에 다운로드하고 import하는 코드입니다.
# 여기서 tensorflow도 같이 import합니다.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 이 코드도 mnist의 데이터를 다운로드하는 역할을 합니다.
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# tensorflow로 계산을 하기 위해 변수들을 정의하는 코드입니다.

# 변수 x는 '숫자가 적힌 이미지'를 의미합니다. [None, 784]는 그 이미지의 규격인데,
# '28*28 픽셀 크기를 가진 숫자(784)들로 이루어진 임의의 자릿수(None)의 숫자'를 뜻합니다.
x = tf.placeholder(tf.float32, [None, 784])

# 변수 W와 b는 이미지에 적힌 숫자들을 추정하는데 단서가 되는 것들입니다.
# 각각 '가중치'와 '편향값'이라고 부릅니다.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 변수 y는 위의 변수 x, W, b와 softmax함수를 사용하여 '모델을 구현하는 역할'을 합니다.
# softmax함수는 입력한 정보가 예상한 정보라고 확신하는 정도를 확률로 나타내는 역할을 합니다.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# '더 나은 답'을 찾기 위한 교차 엔트로피에 필요한 코드입니다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 모델을 학습시키기 전에 변수들을 초기화하고, Session에서 모델을 시작시키는 코드입니다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 모델을 1000번 '학습'시키는 코드입니다.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 예측한 라벨이 맞는 정도를 확인하는 코드입니다.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 예측한 라벨이 맞는 비율을 확인하는 코드입니다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 학습한 모델의 숫자를 맞출 확률을 출력합니다. 이 코드에서는 약 91%입니다.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
