import pandas as pd
import itertools
from collections import defaultdict


class ShapleyValueModel:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def factorial(n):
        if n < 0 or type(n) is not int:
            raise ValueError("n must be integer which is greater than or equal to 0")

        return n * ShapleyValueModel.factorial(n-1) if n != 0 else 1

    @staticmethod
    def powersets(input_list):
        powerset = [list(j) for i in range(len(input_list)) for j in itertools.combinations(input_list, i + 1)]
        return powerset

    @staticmethod
    def subsets(input_list):
        if len(input_list) == 1:
            return input_list
        else:
            sub_list = list()
            for i in range(1, len(input_list)+1):
                sub_list.extend(map(list, itertools.combinations(input_list, i)))

        return list(map(",".join, map(sorted, sub_list)))

    def value_function(self, channel_set, channel_value_info):
        channel_subsets = ShapleyValueModel.subsets(channel_set)
        channel_val = 0
        for subset in channel_subsets:
            if subset in channel_value_info:
                channel_val += channel_value_info[subset]

        return channel_val

    def shapley_value_preprocessing(self):
        self.data['path'] = self.data['path'].apply(lambda x: sorted(set(x)))
        self.data['path'] = self.data['path'].map(tuple)
        self.data = self.data.groupby('path')['total_conversions'].sum().to_frame().reset_index()
        self.data['path'] = self.data['path'].map(list)
        self.data['path'] = self.data['path'].apply(lambda x: ','.join(x))
        return

    def cal_shapley_value(self):
        channel_value_info = self.data.set_index('path').to_dict()['total_conversions']
        self.data['channels'] = self.data['path'].apply(lambda x: x.split(","))
        channels = list(itertools.chain.from_iterable(list(self.data['channels'])))
        channels = list(set(channels))
        num_channel = len(channels)

        channel_set_values = {}
        for channel_set in self.powersets(channels):
            channel_set_values[",".join(sorted(channel_set))] = self.value_function(channel_set, channel_value_info)

        shapley_values = defaultdict(int)
        for channel in channels:
            print(channel)
            for channel_set in channel_set_values.keys():
                if channel not in channel_set.split(","):
                    cardinal_set = len(channel_set.split(","))
                    set_with_channel = channel_set.split(",")
                    set_with_channel.append(channel)
                    set_with_channel = ",".join(sorted(set_with_channel))
                    # Weight = |S|!(n-|S|-1)!/n!, Marginal contribution(i) = value(set U {i})-value(set)
                    weight = (ShapleyValueModel.factorial(cardinal_set)*ShapleyValueModel.factorial(num_channel-cardinal_set-1) / ShapleyValueModel.factorial(num_channel))
                    marginal_contribution = (channel_set_values[set_with_channel]-channel_set_values[channel_set])
                    shapley_values[channel] += weight * marginal_contribution
            shapley_values[channel] += channel_set_values[channel]/num_channel

        shapley_value_attribution = pd.DataFrame(list(shapley_values.items()), columns=['channel', 'attribution'])

        return shapley_value_attribution

    def run(self):
        self.shapley_preprocessing()
        print(self.data)
        shapley_value_attribution = self.cal_shapley_value()

        return shapley_value_attribution


if __name__ == '__main__':
    data = pd.read_csv('sample_data.csv')
    data['path'] = data['path'].apply(lambda x: x.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(','))
    shapley_res = ShapleyValueModel(data=data)

    print(shapley_res.run())


