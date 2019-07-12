start_token = 'G'
end_token = 'E'
import collections
import numpy as np
def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 这里根据包含了每个字对应的频率 Counter({'我': 3, '你': 2, '他': 1})
    counter = collections.Counter(all_words)

    # items转化为列表，里面为元组 [('他', 1), ('你', 2), ('我', 3)]
    count_pairs = sorted(counter.items(), key=lambda x: x[-1])
    # ('他', '你', '我')，(1, 2, 3)
    words, _ = zip(*count_pairs)

    # 取前多少个常用字
    words = words[:len(words)] + (' ',)
    # words = words[:len(words)]
    # 每个字映射为一个数字ID  {'他': 0, '你': 1, '我': 2}
    word_int_map = dict(zip(words, range(len(words))))
    # 将诗歌中每个字转换成一一对应的数字，输入poem中的每个字，转化成对应数字返回
    # 没有获得word对应数字Id,就返回len(words)
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取64首诗进行训练
    # 计算有多少个batch_size
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格进行填充
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches