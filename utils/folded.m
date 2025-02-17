function foldA = folded(unfA, origSize)
    % fold4D: Fold a 3D unfolded array back into its original 4D shape.
    %
    % Inputs:
    %   unfA - The unfolded 3D array.
    %   origSize - A vector representing the original size of the 4D array.
    %
    % Output:
    %   foldA - The folded 4D array.

    % Validate the input dimensions
    if length(origSize) ~= 4
        error('Original size must be a 4D vector.');
    end

    % Extract dimensions from origSize
    [n1, n2, n3, n4] = deal(origSize(1), origSize(2), origSize(3), origSize(4));

    % Initialize the folded array
    foldA = zeros(n1, n2, n3, n4);

    % Perform the folding
    for i = 1:n1
        disp(i)
        for j = 1:n2
            for k = 1:n3
                for l = 1:n4
                    % Calculate the index in the unfolded array
                    idx = (l - 1) * n1 + i;
                    %disp(idx)
                    foldA(i, j, k, l) = unfA(idx, j, k);
                end
            end
        end
    end
end
