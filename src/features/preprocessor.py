#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:51:43 2023

@author: frodo
"""

import pandas as pd

class DataFramePreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe

    def extract_segment(self, s, position):
        segments = s.strip().split('||')
        if position == "first":
            return segments[0]
        elif position == "last":
            return segments[-1]
        else:
            return s

    def parse_timestamp(self, ts):
        date, time_zone = ts.split('T')
        time, time_zone = time_zone.split('.')
        time = time.split('+')[0] if '+' in time else time.split('-')[0] if '-' in time else time
        hour = int(time.split(':')[0])
        return date, time, time_zone, hour

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def convert_to_floats(self, segment_list):
        return [float(val) for val in segment_list if val and self.is_float(val)]

    def compute_sum_for_segment(self, segment_str, separator='|'):
        if not isinstance(segment_str, str):
            print(f"Unexpected type: {type(segment_str)}")
            return segment_str
        segments = segment_str.split(separator)
        return sum(self.convert_to_floats(segments))

    def compute_summation_optimized(self, sum_cols):
        for col in sum_cols:
            self.df[col] = self.df[col].apply(self.compute_sum_for_segment)
        return self.df

    def preprocess(self):
        self.df['segmentsDepartureTimeRaw'] = self.df['segmentsDepartureTimeRaw'].apply(lambda x: self.extract_segment(x, "first"))
        self.df['segmentsDepartureAirportCode'] = self.df['segmentsDepartureAirportCode'].apply(lambda x: self.extract_segment(x, "first"))
        self.df['segmentsArrivalTimeRaw'] = self.df['segmentsArrivalTimeRaw'].apply(lambda x: self.extract_segment(x, "last"))
        self.df['segmentsArrivalAirportCode'] = self.df['segmentsArrivalAirportCode'].apply(lambda x: self.extract_segment(x, "last"))
        self.df['departure_date'], self.df['departure_time'], self.df['departure_time_zone'], self.df['departure_hour'] = zip(*self.df['segmentsDepartureTimeRaw'].apply(self.parse_timestamp))
        self.df['arrival_date'], self.df['arrival_time'], self.df['arrival_time_zone'], self.df['arrival_hour'] = zip(*self.df['segmentsArrivalTimeRaw'].apply(self.parse_timestamp))
        self.df = self.df.drop(columns=['segmentsDepartureTimeRaw','segmentsArrivalTimeRaw',
                                        'segmentsDepartureTimeEpochSeconds','segmentsArrivalTimeEpochSeconds', 
                                        'segmentsAirlineName','arrival_date','arrival_time_zone','arrival_time',
                                        'departure_date','departure_time_zone','departure_time'
                                       ], errors='ignore')
        self.df = self.compute_summation_optimized(['segmentsDurationInSeconds', 'segmentsDistance'])

        return self.df
