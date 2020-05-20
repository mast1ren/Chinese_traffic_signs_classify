import matplotlib.pyplot as plt

class_name = ['speed_Limit_5km/h', 'speed_limit_15km/h', 'speed_limit_30km/h', 'speed_limit_40km/h',
              'speed_limit_50km/h', 'speed_limit_60km/h', 'speed_limit_70km/h', 'speed_limit_80km/h',
              'no_stright_or_left', 'no_stright_or_right', 'no_stright', 'no_left', 'no_left_or_right', 'no_right',
              'no_passing', 'no_u_turn', 'no_vehicles', 'no_horn', 'end_of_speed_limit_40km/h',
              'end_of_speed_limit_50km/h', 'go_stright_or_right', 'go_sright', 'go_left', 'go_left_or_right',
              'go_right', 'keep_left', 'keep_right', 'roundabout_mandatory', 'vehicles', 'horn', 'bicycle', 'u_turn',
              'bypass_on_left_or_right', 'traffic_signal', 'warning', 'pedestrians', 'bicycles_crossing',
              'school_ahead', 'hard_right', 'hard_left', 'downhill_ahead', 'uphill_ahead', 'slow_down',
              'right_T_junction_ahead', 'left_T_junction_ahead', 'village_ahead', 'continuous_hard_turn',
              'unattended_railway_ahead', 'road_work', 'bumpy_road', 'attended_railway_ahead', 'rear-end_attention',
              'stop', 'no_thoroughfare', 'no_parking', 'no_entry', 'yield', 'stop_and_check']


def plotScore(plot_x, plot_acc, plot_recall, plot_precision, plot_f1, plot_x_loss, plot_loss):
    plt.figure(figsize=(10, 10), dpi=100)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.5)
    plt.subplot(grid[0, 0])
    plt.plot(plot_x, plot_acc, 'o-b')
    plt.title('accurary', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accurary', fontsize=14)

    plt.subplot(grid[0, 1])
    plt.plot(plot_x, plot_recall, 'o-b')
    plt.title('recall', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('recall', fontsize=14)

    plt.subplot(grid[1, 0])
    plt.plot(plot_x, plot_precision, 'o-b')
    plt.title('precision', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('precision', fontsize=14)

    plt.subplot(grid[1, 1])
    plt.plot(plot_x, plot_f1, 'o-b')
    plt.ylim(0, 1)
    plt.title('f1', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('f1', fontsize=14)

    plt.subplot(grid[2, 0:2])
    plt.plot(plot_x_loss, plot_loss, 'o-b')
    plt.title('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)

    plt.savefig(fname='./model&img/score.svg', format='svg')

    plt.show()


def plotPredictedResult(img_path, predicted, percent):
    predicted_label = []
    for i in range(5):
        predicted_label.append(class_name[predicted[i]])

    predicted_class = range(len(predicted), 0, -1)

    plt.figure(figsize=(10, 5), dpi=100)
    grid = plt.GridSpec(1, 2, wspace=0.8)

    # image
    image = plt.imread(img_path)
    plt.subplot(grid[0, 0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('image')

    # predicted result
    plt.subplot(grid[0, 1])
    plt.barh(predicted_class, percent, align='center')
    plt.yticks(predicted_class, predicted_label)
    plt.xlim(0, 1)
    plt.title('credibility')
    plt.show()
