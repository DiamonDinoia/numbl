function A = readmatrix(filename, varargin)
    % Parse name-value pairs
    delimiter = '';
    numHeaderLines = -1; % -1 means auto-detect

    i = 1;
    while i <= length(varargin)
        if ischar(varargin{i}) || isstring(varargin{i})
            key = lower(char(varargin{i}));
            if strcmp(key, 'delimiter')
                delimiter = char(varargin{i+1});
                i = i + 2;
            elseif strcmp(key, 'numheaderlines')
                numHeaderLines = varargin{i+1};
                i = i + 2;
            else
                % Skip unknown name-value pairs
                i = i + 2;
            end
        else
            i = i + 1;
        end
    end

    % Auto-detect delimiter from extension if not specified
    if isempty(delimiter)
        [~, ~, ext] = fileparts(filename);
        if strcmp(ext, '.csv')
            delimiter = ',';
        else
            delimiter = ''; % will split on whitespace
        end
    end

    % Read the entire file
    txt = fileread(filename);

    % Split into lines
    lines = strsplit(txt, sprintf('\n'));

    % Remove trailing empty line (from trailing newline)
    if ~isempty(lines) && strcmp(strtrim(char(lines{end})), '')
        lines = lines(1:end-1);
    end

    if isempty(lines)
        A = [];
        return;
    end

    % Auto-detect header lines: skip lines that can't be fully parsed as numbers
    if numHeaderLines < 0
        numHeaderLines = 0;
        for k = 1:length(lines)
            line = strtrim(char(lines{k}));
            if strcmp(line, '')
                numHeaderLines = numHeaderLines + 1;
                continue;
            end
            if ~isempty(delimiter)
                parts = strsplit(line, delimiter);
            else
                parts = strsplit(line);
            end
            allNumeric = true;
            for j = 1:length(parts)
                val = str2double(strtrim(char(parts{j})));
                if isnan(val)
                    token = strtrim(char(parts{j}));
                    % Allow NaN, Inf, -Inf as valid numeric tokens
                    if ~strcmpi(token, 'nan') && ~strcmpi(token, 'inf') && ~strcmpi(token, '-inf')
                        allNumeric = false;
                        break;
                    end
                end
            end
            if allNumeric
                break;
            else
                numHeaderLines = numHeaderLines + 1;
            end
        end
    end

    % Parse data lines
    dataLines = lines(numHeaderLines+1:end);
    nRows = length(dataLines);
    if nRows == 0
        A = [];
        return;
    end

    % First pass: determine number of columns from first data line
    firstLine = strtrim(char(dataLines{1}));
    if ~isempty(delimiter)
        parts = strsplit(firstLine, delimiter);
    else
        parts = strsplit(firstLine);
    end
    nCols = length(parts);

    A = zeros(nRows, nCols);
    for r = 1:nRows
        line = strtrim(char(dataLines{r}));
        if strcmp(line, '')
            A(r, :) = NaN;
            continue;
        end
        if ~isempty(delimiter)
            parts = strsplit(line, delimiter);
        else
            parts = strsplit(line);
        end
        for c = 1:min(length(parts), nCols)
            val = str2double(strtrim(char(parts{c})));
            if isnan(val)
                token = strtrim(char(parts{c}));
                if strcmpi(token, 'nan')
                    A(r, c) = NaN;
                elseif strcmpi(token, 'inf')
                    A(r, c) = Inf;
                elseif strcmpi(token, '-inf')
                    A(r, c) = -Inf;
                else
                    A(r, c) = NaN; % non-numeric data becomes NaN
                end
            else
                A(r, c) = val;
            end
        end
        % Fill missing columns with NaN
        if length(parts) < nCols
            A(r, length(parts)+1:nCols) = NaN;
        end
    end
end
