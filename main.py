import sys
import logging
import analyzer.config as con

formatter = logging.Formatter('%(asctime)s [%(module)14s]' +
                              '[%(levelname)8s] %(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
log.addHandler(console)

file = logging.FileHandler('logfile.log', mode='w')
file.setFormatter(formatter)
log.addHandler(file)


def main():
    config = con.Config()
    config.loop()


if __name__ == '__main__':
    main()
