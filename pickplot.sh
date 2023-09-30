
# opens accuracy and loss plots in Eye of GNOME

for arg in "$@"
do
    fname=$(python3 -c "print(hex(ord('$arg'))[2:])")
    eog "./evaluation-plots/accuracy/result-accuracy-$arg-U+$fname.svg" &
    eog "./evaluation-plots/loss/result-loss-$arg-U+$fname.svg" &
done

