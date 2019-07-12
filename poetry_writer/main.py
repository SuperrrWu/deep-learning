import collections
import os
import sys
import re
import numpy as np
import tensorflow as tf
from model import rnn_model
from poems import process_poems, generate_batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
# set this to 'main.py' relative path
tf.flags.DEFINE_string('checkpoints_dir', './checkpoints/', 'checkpoints save path.')
tf.flags.DEFINE_string('file_path', 'E:/Python/新建文件夹/TensorflowLesson/poetry.txt', 'file name of poems.')

tf.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.flags.FLAGS

start_token = 'G'
end_token = 'E'


# 开始训练
def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)
    # 单词转化的数字:向量，单词和数字一一对应的字典，单词
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # 真实值和目标值
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)
    # 数据占位符
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)
    # 实例化保存模型
    saver = tf.train.Saver(tf.global_variables())
    # 全局变量进行初始化
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        # 先执行，全局变量初始化
        sess.run(init_op)

        start_epoch = 0
        # 把之前训练过的checkpoint拿出来
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            # 拿出训练保存模型
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                # 多少行唐诗//每次训练的个数
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],  # 损失
                        end_points['last_state'],  # 最后一次输出
                        end_points['train_op']  # 训练优化损失
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                if epoch % 6 == 0:  # 每隔多少次保存
                    saver.save(sess, FLAGS.checkpoints_dir, global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, FLAGS.checkpoints_dir, global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    # searchsorted 在前面查找后面的
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    # sample = np.argmax(predict)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


# 调用模型生成诗句
def gen_poem(begin_words, num):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    # 单词转化的数字:向量，单词和数字一一对应的字典，单词
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)
    # 此时输入为1个
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    # 损失等
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # 保存模型的位置，拿回sess
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        # checkpoint = tf.train.latest_checkpoint('./model/')

        saver.restore(sess, checkpoint)
        # saver.restore(sess,'./model/-24')
        # 从字典里面获取到的开始值
        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        poem = ''
        for begin_word in begin_words:

            while True:
                if begin_word:
                    word = begin_word
                else:
                    word = to_word(predict, vocabularies)
                sentence = ''
                while word != end_token:
                    sentence += word
                    x = np.zeros((1, 1))
                    x[0, 0] = word_int_map[word]
                    [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                     feed_dict={input_data: x, end_points['initial_state']: last_state})
                    word = to_word(predict, vocabularies)
                # word = words[np.argmax(probs_)]
                if len(sentence) == 2 + 2 * num and ('，' or '？') not in sentence[:num] and ('，' or '？') not in sentence[
                                                                                                               num + 1:-1] and \
                                sentence[num] == '，' and '□' not in sentence:
                    poem += sentence
                    # sentence = ''
                    break
                else:
                    print("我正在写诗呢")

        return poem


# 这里将生成的诗句，按照中文诗词的格式输出
# 同时方便接入应用
def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    # print(poem_sentences)
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            # if s != '':

            print(s + '。')


def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '1':
            print('[INFO] train tang poem...')
            run_training()
        elif sys.argv[1] == '2':
            num = int(input("请输入训练诗句（5：五言，7：七言):"))
            if num == 5 or num == 7:
                print('[INFO] write tang poem...')
                begin_word = input('开始作诗，请输入起始字:')
                if len(begin_word) == 0:
                    print("请输入词句")
                    return
                r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                begin_word = re.sub(r1, '', begin_word)
                poem2 = gen_poem(begin_word, num)
                pretty_print_poem(poem2)
            else:
                print('输入有误')

        else:
            print('a', sys.argv[1])
            print("请按照以下方式执行：")
            print("python xxxx.py 1(1：训练，2：写诗)")
    else:
        print(len(sys.argv))
        print("请按照以下方式执行：")
        print("python xxxx.py 1(1：训练，2：写诗)")


if __name__ == '__main__':
    main()