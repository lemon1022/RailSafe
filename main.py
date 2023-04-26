from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.CustomMessageBox import MessageBox
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_imshow, set_logging, increment_path, \
    polygon_non_max_suppression, polygon_scale_coords
from utils.plots import polygon_plot_one_box, colors

from utils.torch_utils import select_device, time_synchronized
from dialog.rtsp_win import Window


class Point():
    def __init__(self, x, y):
         self.x = x
         self.y = y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)


def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn


def check_coin(person_row, line_row):
    x1 = person_row[6].item()
    y1 = person_row[7].item()
    x2 = person_row[4].item()
    y2 = person_row[5].item()

    axis_line = []
    for i in range(0, len(line_row)-2, 2):
        axis_line.append((line_row[i].item(), line_row[i+1].item()))
    p1 = Point(axis_line[0][0], axis_line[0][1])
    p2 = Point(axis_line[1][0], axis_line[1][1])
    p3 = Point(axis_line[2][0], axis_line[2][1])
    p4 = Point(axis_line[3][0], axis_line[3][1])
    pp = Point(x1, y1)
    pp2 = Point(x2, y2)

    if p4.x != p1.x:
        k1 = (p4.y - p1.y) / (p4.x - p1.x)
        b1 = p1.y - k1 * p1.x
        res1 = k1 * pp.x + b1 - pp.y
        res1 *= k1
    else:
        if pp.x < p1.x:
            res1 = -1
        elif pp.x == p1.x:
            res1 = 0
        else:
            res1 = 1

    if p3.x != p2.x:
        k2 = (p3.y - p2.y) / (p3.x - p2.x)
        b2 = p2.y - k2 * p2.x
        res2 = k2 * pp2.x + b2 - pp2.y
        res2 *= k2
    else:
        if pp2.x < p1.x:
            res2 = -1
        elif pp2.x == p1.x:
            res2 = 0
        else:
            res2 = 1

    ok_1 = IsPointInMatrix(p1, p2, p3, p4, pp)
    ok_2 = IsPointInMatrix(p1, p2, p3, p4, pp2)

    return ok_1 | ok_2 | (res1 * res2 <= 0)


def deal(det):
    del_row = []
    for i, det_row in enumerate(det):
        if det_row[-1] == 1:
            danger_person = False
            for j, row in enumerate(det):
                if row[-1] == 0:
                    coin = check_coin(det_row, row)
                    if coin:
                        danger_person = True
            if not danger_person:
                del_row.append(i)

    if len(del_row):
        det = np.delete(det, del_row, 0)
    # print(det)
    return det


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)
# E:\pic\\test01.mp4
    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './pt/yolov5s.pt'
        self.current_weight = './pt/yolov5s.pt'
        self.source = 'D:\mypro\PyQt5-YOLOv5-yolov5_v6.1\data\images\\test.jpg'
        self.conf_thres = 0.65
        self.iou_thres = 0.65
        self.jump_out = False  # jump out of the loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'

    @torch.no_grad()
    def run(self,
               imgsz=640,  # inference size (pixels)
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               view_img=True,  # show results
               save_txt=False,  # save results to *.txt
               save_conf=False,  # save confidences in --save-txt labels
               save_crop=False,  # save cropped prediction boxes
               nosave=False,  # do not save images/videos
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               update=False,  # update all models
               project='runs/detect',  # save results to project/name
               name='exp',  # save results to project/name
               exist_ok=False,  # existing project/name ok, do not increment
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               half=False,  # use FP16 half-precision inference
               ):

        save_img = not nosave and not self.source.endswith('.txt')  # save inference images
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Polygon does not support second-stage classifier
        classify = False
        assert not classify, "polygon does not support second-stage classifier"

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        count = 0

        while True:
            if self.jump_out:
                self.vid_cap.release()
                self.send_percent.emit(0)
                self.send_msg.emit('Stop')
                if hasattr(self, 'out'):
                    self.out.release()
                break
            if self.is_continue:
                path, img, im0s, self.vid_cap = next(dataset)

                # path, img, im0s, self.vid_cap =
                count += 1
                if count % 30 == 0 and count >= 30:
                    fps = int(30 / (time.time() - t0))
                    self.send_fps.emit('fps：' + str(fps))
                    t0 = time.time()
                if self.vid_cap:
                    percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                    self.send_percent.emit(percent)
                else:
                    percent = self.percent_length

                statistic_dic = {name: 0 for name in names}

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=augment)[0]

                # Apply polygon NMS
                pred = polygon_non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path

                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain xyxyxyxy

                    assert not save_crop, "polygon does not support save_crop"
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], im0.shape).round()
                        # Print results
                        # print(det)
                        det = deal(det)
                        # print(det)

                        for c in det[:, -1].unique():
                            # print(det[:, -1], c)
                            n = (det[:, -1] == c).sum()  # detections per class
                            statistic_dic[names[int(c)]] += 1
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxyxyxy, conf, cls in reversed(det):
                            if save_img or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                polygon_plot_one_box(torch.tensor(xyxyxyxy).cpu().numpy(), im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                if self.rate_check:
                    time.sleep(1 / self.rate)

                self.send_img.emit(im0)
                self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                self.send_statistic.emit(statistic_dic)

                if percent == self.percent_length:
                    print(count)
                    self.send_percent.emit(0)
                    self.send_msg.emit('finished')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break



class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1:窗口可以拉伸
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
        #
        # # style 2: 窗口不能拉伸
        # self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
        #                     | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # 窗口透明度

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # 显示最大化窗口
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))

        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None


    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)

        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        # self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()
        # print(self.det_thread.source)


    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    # det = DetThread()
    # det.run()
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
