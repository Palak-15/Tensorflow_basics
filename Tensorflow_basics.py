import tensorflow as tf

node1=tf.constant(3.0,dtype=tf.float32)
node2=tf.constant(4.0)
print(node1,node2)

sess=tf.Session()
print(sess.run([node2,node1]))
print(sess.run(tf.add(node1,node2)))


a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder=a+b
print(sess.run(adder,{a:2,b:4.5}))
mul=adder*3
print(sess.run(mul,{a:4,b:10}))

m=tf.Variable([.3],dtype=tf.float32)
c=tf.Variable([-.3],dtype=tf.float32)
x=tf.placeholder(tf.float32)
y=m*x +c

init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(y,{x:[1,2,3,4,5]}))
actual=tf.placeholder(tf.float32)
squared_deltas=tf.square(y-actual)
loss=tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4],actual:[0,-1,2,3]}))

fix=tf.assign(m,[1.])
sess.run([fix,c])
print(sess.run(loss,{x:[1,2,3,4],actual:[0,-1,2,3]}))


optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
	sess.run(train,{x:[1,2,3,4] ,actual:[0,-1,-2,1]})
print(sess.run([m,c]))

