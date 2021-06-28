import pandas as pd
import numpy as np


class MarkovChainAttribution:
    def __init__(self, data, null_exists):
        """
        :param data: a dataframe with path, total_conversions columns
        :param null_exists: whether data's path only have conversion or have conversion and null
        """
        self.data = data
        self.null_exists = null_exists

    @staticmethod
    def list_to_char(input_list):
        output_str = ""
        for idx, element in enumerate(input_list):
            if idx == 0:
                output_str = str(element).replace("'", '')
            else:
                output_str = output_str + ' > ' + str(element).replace("'", '')
        return output_str

    def markov_chain_preprocessing(self):
        """ Markov chain attribution model have single path problem. While calculating removal_effects, single path lose
        its own conversions to other channels. Assume single path 'start -> A -> A -> Conversion : 11' In markov chain
        attribution, A cannot take whole 11 conversions. Finally single channel's are undervalued and attribution distorted.
        Therefore, split data to single, multi channel. And run markov chain model for only multichannel data.
        :return: singlechannel data, multichannel data
        """

        self.data['Unique Path Num'] = self.data['path'].apply(lambda x: len(list(set(x))))
        singlechnl_df = self.data.loc[self.data['Unique Path Num'] == 1]
        singlechnl_df['path'] = singlechnl_df['path'].apply(lambda x: x[0])
        singlechnl_df = singlechnl_df.drop(['Unique Path Num'], axis=1)

        multichnl_df = self.data.loc[self.data['Unique Path Num'] != 1]
        multichnl_df = multichnl_df.drop(['Unique Path Num'], axis=1)
        multichnl_df['path'].apply(lambda x: x.insert(0, 'start'))
        multichnl_df['path'].apply(lambda x: x.append('conv'))
        multichnl_df['path'] = multichnl_df['path'].apply(self.list_to_char)
        multichnl_df = multichnl_df.reindex(multichnl_df.index.repeat(multichnl_df.total_conversions)) #for kolon's data.
        multichnl_df = multichnl_df.drop(['total_conversions'], axis=1)
        multichnl_df.columns = ['Paths']

        if not self.null_exists:
            null_row = {'Paths': 'start > Kobby > Jayden > null'}
            multichnl_df = multichnl_df.append(null_row, ignore_index=True)

        return singlechnl_df, multichnl_df

    def cal_removal_effect(self, df, cvr):
        """ Calculate removal effect. Removal Effect(i) = {1-P(S without i)/P(S)}*100 (%)
        In this code, solve Loop Issue by using markov matrix's stability. Due to markov matrix's stability, we can
        assume steady state.
        :param df: channel data
        :param cvr: conversion rate
        :return:
        """
        removal_effect_res = dict()
        channels = df.drop(['conv', 'null', 'start'], axis=1).columns
        for channel in channels:
            removal_df = df.drop(channel, axis=1)
            removal_df = removal_df.drop(channel, axis=0)
            for col in removal_df.columns:
                trans_prob_sum = np.sum(list(removal_df.loc[col]))
                null_prob = float(1) - trans_prob_sum
                if null_prob == 0:
                    continue
                else:
                    removal_df.loc[col]['null'] = null_prob
            removal_df.loc['null']['null'] = 1.0
            R = removal_df[['null', 'conv']]
            R = R.drop(['null', 'conv'], axis=0)
            Q = removal_df.drop(['null', 'conv'], axis=1)
            Q = Q.drop(['null', 'conv'], axis=0)
            t = len(Q.columns)
            # Markov Matrix's absolute stability -> steady state
            N = np.linalg.inv(np.identity(t) - np.asarray(Q))
            M = np.dot(N, np.asarray(R))

            removal_cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
            removal_effect = 1.0 - removal_cvr / cvr
            removal_effect_res[channel] = removal_effect

        return removal_effect_res


    def cal_markov_chain_attribution(self, multichnl_df):
        """ Calculate markov chain attribution
        :param multichnl_df: multichannel path data
        :return: markov chain attribution data
        """
        paths = np.array(multichnl_df).tolist()
        sublist = []
        total_paths = 0
        for path in paths:
            for touchpoint in path:
                userpath = touchpoint.split(' > ')
                sublist.append(userpath)
            total_paths += 1
        paths = sublist

        unique_touch_list = set(x for element in paths for x in element)
        conv_dict = {}
        total_conversions = 0
        for item in unique_touch_list:
            conv_dict[item] = 0
        for path in paths:
            if 'conv' in path:
                total_conversions += 1
                conv_dict[path[-2]] += 1

        transitionStates = {}
        for x in unique_touch_list:
            for y in unique_touch_list:
                transitionStates[x + ">" + y] = 0

        for possible_state in unique_touch_list:
            if possible_state != "null" and possible_state != "conv":
                for user_path in paths:
                    if possible_state in user_path:
                        indices = [i for i, s in enumerate(user_path) if possible_state == s]
                        for col in indices:
                            transitionStates[user_path[col] + ">" + user_path[col + 1]] += 1

        transitionMatrix = []
        actual_paths = []
        for state in unique_touch_list:
            if state != "null" and state != "conv":
                counter = 0
                index = [i for i, s in enumerate(transitionStates) if s.startswith(state + '>')]
                for col in index:
                    if transitionStates[list(transitionStates)[col]] > 0:
                        counter += transitionStates[list(transitionStates)[col]]
                for col in index:
                    if transitionStates[list(transitionStates)[col]] > 0:
                        state_prob = float((transitionStates[list(transitionStates)[col]])) / float(counter)
                        actual_paths.append({list(transitionStates)[col]: state_prob})
        transitionMatrix.append(actual_paths)

        flattened_matrix = [item for sublist in transitionMatrix for item in sublist]
        transState = []
        transMatrix = []
        for item in flattened_matrix:
            for key in item:
                transState.append(key)
            for key in item:
                transMatrix.append(item[key])

        tmatrix = pd.DataFrame({'paths': transState,
                                'prob': transMatrix})
        tmatrix = tmatrix.join(tmatrix['paths'].str.split('>', expand=True).add_prefix('channel'))[
            ['channel0', 'channel1', 'prob']]
        column = list()
        for k, v in tmatrix.iterrows():
            if v['channel0'] in column:
                continue
            else:
                column.append(v['channel0'])
        test_df = pd.DataFrame()
        for col in unique_touch_list:
            test_df[col] = 0.00
            test_df.loc[col] = 0.00
        for k, v in tmatrix.iterrows():
            x = v['channel0']
            y = v['channel1']
            val = v['prob']
            test_df.loc[x][y] = val
        test_df.loc['conv']['conv'] = 1.0
        test_df.loc['null']['null'] = 1.0
        R = test_df[['null', 'conv']]
        R = R.drop(['null', 'conv'], axis=0)
        Q = test_df.drop(['null', 'conv'], axis=1)
        Q = Q.drop(['null', 'conv'], axis=0)
        O = pd.DataFrame()
        t = len(Q.columns)
        for col in range(0, t):
            O[col] = 0.00
        for col in range(0, len(R.columns)):
            O.loc[col] = 0.00
        N = np.linalg.inv(np.identity(t) - np.asarray(Q))
        M = np.dot(N, np.asarray(R))
        cvr = pd.DataFrame(M, index=R.index)[[1]].loc['start'].values[0]
        removal_effects = self.cal_removal_effect(test_df, cvr)
        denominator = np.sum(list(removal_effects.values()))
        allocation_amount = list()
        for i in removal_effects.values():
            allocation_amount.append((i / denominator) * total_conversions)
        markov_conversions = dict()
        i = 0
        for channel in removal_effects.keys():
            markov_conversions[channel] = allocation_amount[i]
            i += 1
        conv_dict.pop('conv', None)
        conv_dict.pop('null', None)
        conv_dict.pop('start', None)

        return markov_conversions


    def run(self):
        singlechnl_df, multichnl_df = self.markov_chain_preprocessing()
        multichnl_markov_res = self.cal_markov_chain_attribution(multichnl_df)
        multichnl_attr_df = pd.DataFrame.from_dict(multichnl_markov_res, orient='index').reset_index()
        markov_chain_attribution = pd.concat([singlechnl_df, multichnl_attr_df], sort=False)
        markov_chain_attribution = markov_chain_attribution.groupby('path')['total_conversions'].sum().to_frame()
        markov_chain_attribution['total_conversions'] = markov_chain_attribution['total_conversions'].map(int)
        markov_chain_attribution = markov_chain_attribution.sort_values(by=['total_conversions'], ascending=False)

        return markov_chain_attribution

if __name__ == '__main__':
    data = pd.read_csv('sample_data.csv')
    data['path'] = data['path'].apply(lambda x: x.replace("'", "").replace('[', '').replace(']', '').replace(' ', '').split(','))

    markov_chain_attribution = MarkovChainAttribution(data=data, null_exists=False)
    markov_chain_res = markov_chain_attribution.run()
    print(markov_chain_res)
