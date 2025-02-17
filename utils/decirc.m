function A = decirc(circ_A, origSize)
    % decirc: Reconstruct the original 4th order tensor from its block-circulant matrix.
    %
    % Inputs:
    %   circ_A - The block-circulant matrix.
    %   origSize - A vector representing the original size of the 4th order tensor.
    %
    % Output:
    %   A - The reconstructed 4th order tensor.

    % Validate the original size
    if length(origSize) ~= 4
        error('Original size must be a 4D vector.');
    end

    % Extract original dimensions
    n2 = origSize(2);
    % Initialize the output tensor
    
    A = folded(circ_A(:,1:n2,:),origSize);
end
