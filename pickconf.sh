
# opens only loss plots in Eye of GNOME

for arg in "$@"
do
    fname=$(python3 -c "print(hex(ord('$arg'))[2:])")
    eog "./evaluation-plots/loss/confusionmatrix-$arg-U+$fname.svg" &
done

