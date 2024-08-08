lamb = 0.76;
res = 96;
load("library/pillars - supercell/z_real_half_" + num2str(lamb) + "um.mat", 'z_real');
load("library/pillars - supercell/z_imag_half_" + num2str(lamb) + "um.mat", 'z_imag');

z_imag = fill(z_imag, res);
z_real = fill(z_real, res);

save("library/pillars - supercell/z_real_" + num2str(lamb) + "um.mat", 'z_real');
save("library/pillars - supercell/z_imag_" + num2str(lamb) + "um.mat", 'z_imag');

function arr = fill(half_arr, N)

    arr = zeros(N, N);
    counter = 1;
    for i = 1:N
        for j = 1:N
            if j >= i
                arr(i, j) = half_arr(counter);
                counter = counter + 1;
            else
                arr(i, j) = arr(j, i);
            end
        end
    end
end
