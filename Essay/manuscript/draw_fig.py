
from data_smoothed import smooth
from matplotlib import pyplot as plt


nr_psnr = "../logs/plot_data/IRSRGAN_org_no_pretrained(random_deg)/PSNR.csv"
wnr_psnr = "../logs/plot_data/IRSRGAN_org_with_pretrained/PSNR.csv"

x, y = smooth(nr_psnr)


plt.plot(x, y)
plt.show()