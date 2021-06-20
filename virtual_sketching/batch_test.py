from virtual_sketching.test_vectorization import *
import time


def main_self(model_name, test_image_name, sampling_num):
    test_dataset = 'clean_line_drawings'
    test_image_base_dir = '/home/tan/POSTGRADUATE/sketch_simplification-master-server/data/'
    # test_image_base_dir = 'C:/Users/tan/Desktop/data/'

    sampling_base_dir = 'outputs/sampling'
    model_base_dir = 'outputs/snapshot'

    state_dependent = False
    longer_infer_lens = [700 for _ in range(10)]
    round_stop_state_num = 12
    stroke_acc_threshold = 0.95

    draw_seq = False
    draw_color_order = True

    # set numpy output to something sensible
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    main_testing(test_image_base_dir, test_dataset, test_image_name,
                 sampling_base_dir, model_base_dir, model_name, sampling_num,
                 draw_seq=draw_seq, draw_order=draw_color_order,
                 state_dependent=state_dependent, longer_infer_lens=longer_infer_lens,
                 round_stop_state_num=round_stop_state_num, stroke_acc_threshold=stroke_acc_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', help="The test image name.")
    parser.add_argument('--model', '-m', type=str, default='pretrain_clean_line_drawings', help="The trained model.")
    parser.add_argument('--sample', '-s', type=int, default=6, help="The number of outputs.")
    args = parser.parse_args()

    # args.input = 'C:/Users/tan/Desktop/data/clean_line_drawings/'
    args.input = '/home/tan/POSTGRADUATE/sketch_simplification-master-server/data/clean_line_drawings/'

    assert args.input != ''
    assert args.sample > 0

    item_name_list = os.listdir(args.input)
    item_name_list.sort()
    num = len(item_name_list)
    resume = 273

    start_time = time.time()
    for i in range(273, num):
        name = item_name_list[i]
        print('\n===========================================')
        print('process image {:d}/{:d}, {}'.format(i, num, item_name_list[i]))
        print('===========================================')
        print('Time cost so far: {:.3f}s'.format(time.time() - start_time))

        main_self(args.model, name, args.sample)
