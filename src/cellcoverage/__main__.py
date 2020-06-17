from cellcoverage.analyzers import ParameterTester

def main():
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    an = ParameterTester(outPath=r'G:\HCT116 coverage cele synergy (10-2-19)\1\Ana',
                                    wellPath=r'G:\HCT116 coverage cele synergy (10-2-19)\1\BottomLeft_1',
                                    ffcPath=r'G:\HCT116 coverage cele synergy (10-2-19)\FFC\BottomLeft_1',
                                    darkCount=624,
                                    rotate90=1)
    mask = an.selectAnalysisArea()
    an.run(mask)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()