# USAGE:
# python main.py -p "./single/1M 1.jpg" -s yes
import cv2
import imutils
import numpy as np
import argparse
import csv


def main():
  path, save = get_path_from_args()
  card_id = get_card_id_from_path(path)
  info = get_standard_card_info(card_id)

  img = cv2.imread(path)
  card = get_roi(img)
  bgr = info['b'], info['g'], info['r']
  corrected = correct(card, card, bgr)

  lab_standard = info['l_star'], info['a_star'], info['b_star']
  lab_standard = np.array(lab_standard, np.float64)
  lab_before = calculate_lab_mean(card, two_decimals=False)
  lab_after = calculate_lab_mean(corrected, two_decimals=False)

  store_results_in_csv(card_id, lab_standard, lab_before, lab_after) if boolean(save) else print_results_on_console(
      card_id, lab_standard, lab_before, lab_after)


def boolean(save):
  return save == 'yes' or save == 'y'


def get_path_from_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('-p', '--path', required=True)
  ap.add_argument('-s', '--save', required=False,
                  default='yes', choices=('yes', 'no', 'y', 'n'))
  args = vars(ap.parse_args())

  return args['path'], args['save']


def get_card_id_from_path(path):
  from pathlib import Path
  return Path(path).stem


def get_standard_card_info(card_id):
  with open('3d_master.csv') as file:
    reader = csv.DictReader(file)
    for row in reader:
      if row['card_id'] == card_id:
        return row


def get_roi(img):
  img = imutils.resize(img, 1500)
  x0, y0, dx, dy = cv2.selectROI(
      'select roi', img, showCrosshair=False)
  x1 = x0 + dx
  y1 = y0 + dy
  return img[y0:y1, x0:x1]


def get_mask(img_with_black_bg, threshold_low=50):
  gray = cv2.cvtColor(img_with_black_bg, cv2.COLOR_BGR2GRAY)
  t, mask = cv2.threshold(gray, threshold_low, 255, cv2.THRESH_BINARY)

  return mask


def calculate_lab_mean(img_with_black_bg, show_mask=True, two_decimals=True):
  mask = get_mask(img_with_black_bg, 50)
  if show_mask:
    cv2.imshow('mask when calculating lab mean', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
  lab = cv2.cvtColor(img_with_black_bg, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.mean(lab, mask)[0:3]
  lab = l * 100 / 255, a - 128, b - 128
  return np.around(lab, decimals=2) if two_decimals else lab


def correct(img, card_with_black_bg, bgr_mean_std):
  mask = get_mask(card_with_black_bg, 50)

  bgr_mean_src = cv2.mean(card_with_black_bg, mask)[:3]

  bgr_mean_src = np.array(bgr_mean_src, np.float64)
  bgr_mean_std = np.array(bgr_mean_std, np.float64)

  k_b, k_g, k_r = bgr_mean_std / bgr_mean_src
  b, g, r = cv2.split(img)
  bgr_corrected = b * k_b, g * k_g, r * k_r
  bgr_corrected = np.clip(bgr_corrected, 0, 255)

  return cv2.merge(np.array(bgr_corrected, np.uint8))


def calculate_delta_e(lab_x, lab_y, two_decimals=True):
  delta_e = np.linalg.norm(
      np.array(lab_x, np.float64) - np.array(lab_y, np.float64))
  return np.around(delta_e, decimals=2) if two_decimals else delta_e


def print_results_on_console(card_id, lab_standard, lab_before, lab_after):
  print('------------------------------------------------------')
  print('CARD:', card_id)
  print('------------lab------------')
  print('lab standard:', tuple(np.around(lab_standard, decimals=2)))
  print('lab before:', tuple(np.around(lab_before, decimals=2)))
  print('lab after:', tuple(np.around(lab_after, decimals=2)))
  print('----------delta e----------')
  print('delta e before:',
        calculate_delta_e(lab_before, lab_standard))
  print('delta e after:',
        calculate_delta_e(lab_after, lab_standard))
  print('delta e between before and after:',
        calculate_delta_e(lab_after, lab_before))
  print('---the best matched card---')
  print('before:', get_the_best_matched_card_id(lab_before))
  print('after:', get_the_best_matched_card_id(lab_after))
  print('------------------------------------------------------')


def store_results_in_csv(card_id, lab_standard, lab_before, lab_after):
  delta_e_before = calculate_delta_e(lab_before, lab_standard)
  delta_e_after = calculate_delta_e(lab_after, lab_standard)
  delta_e_between_before_and_after = calculate_delta_e(lab_after, lab_before)
  best_matched_card_before = get_the_best_matched_card_id(lab_before)
  best_matched_card_after = get_the_best_matched_card_id(lab_after)

  lab_standard = tuple(np.around(lab_standard, decimals=2))
  lab_before = tuple(np.around(lab_before, decimals=2))
  lab_after = tuple(np.around(lab_after, decimals=2))

  with open('results.csv', 'a', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    if isEmpty('results.csv'):
      writer.writerow(["card_id", "lab_standard", "lab_before", "lab_after", "delta_e_before", "delta_e_after",
                       "delta_e_between_before_and_after", "best_matched_card_before", "best_matched_card_after"])
    writer.writerow([card_id, lab_standard, lab_before, lab_after, delta_e_before, delta_e_after,
                     delta_e_between_before_and_after, best_matched_card_before, best_matched_card_after])


def isEmpty(filepath):
  import os
  return os.stat(filepath).st_size == 0


def get_the_best_matched_card_id(lab):
  min_delta_e = 1000
  best_card_id = '0M 3'

  with open('3d_master.csv') as file:
    reader = csv.DictReader(file)
    for row in reader:
      lab_row = row['l_star'], row['a_star'], row['b_star']
      delta_e = calculate_delta_e(lab, lab_row, two_decimals=False)
      if delta_e < min_delta_e:
        min_delta_e = delta_e
        best_card_id = row['card_id']

  return best_card_id


if __name__ == '__main__':
  main()
