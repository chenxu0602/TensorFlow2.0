
from person_pb2 import Person

person = Person(name="A1", id=123, email=["a@b.com"])
print(person)

person.email.append("c@d.com")
s = person.SerializeToString()

person2 = Person()
person2.ParseFromString(s)

print(person == person2)


#from features_pb2 import BytesList, FloatList, Int64List
#from features_pb2 import Feature, Features, Example
import tensorflow as tf

person_example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"Alice"])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[123])),
            "emails": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"a@b.com", b"c@d.com"]))
        }
    )
)

with tf.io.TFRecordWriter("my_contracts.tfrecord") as f:
    f.write(person_example.SerializeToString())

feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

for serialized_example in tf.data.TFRecordDataset(["my_contracts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

print(tf.sparse.to_dense(parsed_example["emails"]))
print(parsed_example["emails"].values)

dataset = tf.data.TFRecordDataset(["my_contracts.tfrecord"]).batch(10)
for serialized_example in dataset:
    parsed_example = tf.io.parse_example(serialized_example, feature_description)

"""
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
                                            serialized_sequence_example, 
                                            context_feature_descriptions,
                                            sequence_feature_description)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
"""