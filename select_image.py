import os
import cv2
import pathlib


IMAGE_SIZE = [1224, 1024]
date = 20200731
time = 162309
index = ['1-2-50']
dst_root = 'E:/Image/SaveImage/'
# if station1
#
# image_path = root+str(date)+'/'+str(time)+'/'+'Statoin'+str(station)+'/'+'Line'+str(line)+'/'+
while True:
    date = input('Please input date: ')
    if date == '':
        break
    else:
        while True:
            time = input('Please input time: ')

            if time == '':
                break
            else:
                while True:

                    station = input('Please input station: ')
                    if station == '':
                        break
                    else:
                        while True:
                            line = input('Please input line:')
                            if line == '':
                                break
                            else:
                                while True:
                                    num = input('Please input num: ')
                                    if num == '':
                                        break
                                    else:
                                        if station == '1':
                                            root = 'D:/Image/SaveImage/'
                                            image_path = root + str(date) + '/' + str(
                                                time) + '/' + 'Station1' + '/' + 'Line' + str(line) + '/' + str(
                                                num) + '_up.bmp'
                                            image11 = cv2.imread(image_path, 0)
                                            image1 = cv2.resize(image11, IMAGE_SIZE)
                                            cv2.imshow(image_path, image1)
                                            cv2.waitKey()
                                            image_path = root + str(date) + '/' + str(
                                                time) + '/' + 'Station1' + '/' + 'Line' + str(line) + '/' + str(
                                                num) + '_dn.bmp'
                                            image12 = cv2.imread(image_path, 0)
                                            image2 = cv2.resize(image12, IMAGE_SIZE)
                                            cv2.imshow(image_path, image2)
                                            cv2.waitKey()
                                            image_path = root + str(date) + '/' + str(
                                                time) + '/' + 'Station1' + '/' + 'Line' + str(line) + '/' + str(
                                                num) + '_1.bmp'
                                            image13 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image13)
                                            cv2.waitKey()
                                            image_path = root + str(date) + '/' + str(
                                                time) + '/' + 'Station1' + '/' + 'Line' + str(line) + '/' + str(
                                                num) + '_2.bmp'
                                            image14 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image14)
                                            cv2.waitKey()
                                            cv2.destroyAllWindows()

                                            defect = input('Please input type: ')
                                            if defect != '':
                                                dst_root = 'E:/Images/' + defect + '/'
                                                if not os.path.exists(dst_root):
                                                    os.makedirs(dst_root)
                                                if defect == 'surplus':
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-1-' + str(line) + str(
                                                            num) + '_1.bmp', image13)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-1-' + str(line) + str(
                                                            num) + '_2.bmp', image14)
                                                elif defect == 'lack':
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-1-' + str(line) + str(
                                                            num) + '_up.bmp', image11)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-1-' + str(line) + str(
                                                            num) + '_dn.bmp', image12)

                                            else:
                                                print('no spcific type insert, turn to station4')
                                                image_path = root + str(date) + '/' + str(
                                                    time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                    num) + '_dn.bmp'
                                                image11 = cv2.imread(image_path, 0)
                                                image1 = cv2.resize(image11, IMAGE_SIZE)
                                                cv2.imshow(image_path, image1)
                                                cv2.waitKey()
                                                image_path = root + str(date) + '/' + str(
                                                    time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                    num) + '_mid.bmp'
                                                image12 = cv2.imread(image_path, 0)
                                                image2 = cv2.resize(image12, IMAGE_SIZE)
                                                cv2.imshow(image_path, image2)
                                                cv2.waitKey()
                                                image_path = root + str(date) + '/' + str(
                                                    time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                    num) + '_up.bmp'
                                                image13 = cv2.imread(image_path, 0)
                                                image3 = cv2.resize(image13, IMAGE_SIZE)
                                                cv2.imshow(image_path, image3)
                                                cv2.waitKey()
                                                cv2.destroyAllWindows()

                                                defect = input('Please input type: ')
                                                if defect != '':
                                                    dst_root = 'E:/Images/' + defect + '/'
                                                    if not os.path.exists(dst_root):
                                                        os.makedirs(dst_root)
                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_dn_1.bmp'
                                                    image41 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_dn_1.bmp', image41)
                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_dn_2.bmp'
                                                    image42 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_dn_2.bmp', image42)

                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_mid_1.bmp'
                                                    image43 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_mid_1.bmp', image43)

                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_mid_2.bmp'
                                                    image44 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_mid_2.bmp', image44)

                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_up_1.bmp'
                                                    image45 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_up_1.bmp', image45)

                                                    image_path = root + str(date) + '/' + str(
                                                        time) + '/' + 'Station4' + '/' + 'Line' + str(line) + '/' + str(
                                                        num) + '_up_2.bmp'
                                                    image46 = cv2.imread(image_path, 0)
                                                    cv2.imwrite(
                                                        dst_root + str(date) + '-' + str(time) + '-4-' + str(
                                                            line) + str(
                                                            num) + '_up_2.bmp', image46)


                                        elif station == '2':
                                            root = 'E:/Image/SaveImage/'
                                            image_path = root+str(date)+'/'+str(time)+'/'+'Station2'+'/'+'Line'+str(line)+'/'+str(num)+'_1.bmp'
                                            image21 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image21)
                                            cv2.waitKey()
                                            image_path = root+str(date)+'/'+str(time)+'/'+'Station2'+'/'+'Line'+str(line)+'/'+str(num)+'_2.bmp'
                                            image22 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image22)
                                            cv2.waitKey()
                                            image_path = root+str(date)+'/'+str(time)+'/'+'Station3'+'/'+'Line'+str(line)+'/'+str(num)+'_1.bmp'
                                            image31 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image31)
                                            cv2.waitKey()
                                            image_path = root+str(date)+'/'+str(time)+'/'+'Station3'+'/'+'Line'+str(line)+'/'+str(num)+'_2.bmp'
                                            image32 = cv2.imread(image_path, 0)
                                            cv2.imshow(image_path, image32)
                                            cv2.waitKey()
                                            cv2.destroyAllWindows()

                                            defect = input('Please input type: ')
                                            if defect != '':

                                                dst_root = 'E:/IMAGES/'+defect+'/'

                                                if not os.path.exists(dst_root):
                                                    os.makedirs(dst_root)
                                                cv2.imwrite(dst_root+str(date)+'-'+str(time)+'-2-'+str(line)+str(num)+'_1.bmp', image21)
                                                cv2.imwrite(
                                                    dst_root + str(date) + '-' + str(time) + '-2-' + str(line) + str(
                                                        num) + '_2.bmp', image22)
                                                cv2.imwrite(
                                                    dst_root + str(date) + '-' + str(time) + '-3-' + str(line) + str(
                                                        num) + '_1.bmp', image31)
                                                cv2.imwrite(
                                                    dst_root + str(date) + '-' + str(time) + '-3-' + str(line) + str(
                                                        num) + '_2.bmp', image32)


