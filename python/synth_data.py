from random import randint
import sys


def main():
    print str(sys.argv)
    dim = 0
    num_of_instances = 0
    try:
        dim = int(sys.argv[1])
        num_of_instances = int(sys.argv[2])
    except Exception as e:
        print "Failed to read args: %s" % e

    with open("synth_data_instances.txt", "w+") as f_instances:
        with open("synth_data_labels.txt", "w+") as f_labels:
            for i in xrange(num_of_instances):
                positive_index = randint(0, dim-1)
                cur_vec = [0] * dim
                cur_vec[positive_index] = 1
                f_instances.write(str(cur_vec) + '\n')
                f_labels.write(str(positive_index) + '\n')


if __name__ == "__main__":
    main()
