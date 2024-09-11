file_name = 'aux_dynamic_squat_0kg.mat'
indices = [9, 10, 11, 12, 15, 16, 60, 61, 62, 63]
% Get the current directory
base_dir = '/home/rwalia/Downloads/OneDrive_1_9-10-2024/markerdata/';

% Create a new directory with the same name as the input file
output_dir = fullfile(base_dir, lower(file_name(1:end-4))); % remove '.mat' extension for the folder
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Iterate over the subdirectories (SUB1 to SUB9)
sub_dirs = [1:4, 6:10]; % Exclude 5
for j = 1:length(sub_dirs)
    i = sub_dirs(j); % Current subdirectory number
    sub_dir = fullfile(base_dir, ['SUB', num2str(i)]);
    mat_files = dir(fullfile(sub_dir, '*.mat'));

    mat_file_found = false;
    for k = 1:length(mat_files)
        if strcmpi(mat_files(k).name, file_name)
            mat_file = fullfile(sub_dir, mat_files(k).name);
            mat_file_found = true;
            actual_field_name = mat_files(k).name(1:end-4);
            break;
        end
    end

    if mat_file_found
        loaded_data = load(mat_file);
        fields = fieldnames(loaded_data);
        field_found = false;
        for k = 1:length(fields)

        % Access the trajectory data
            if isfield(loaded_data, actual_field_name)
              traj = loaded_data.(actual_field_name).Trajectories;
              labeled_data = traj.Labeled;
              data = labeled_data.Data;
              labels = labeled_data.Labels;

              % Initialize matrix to store extracted data for the current SUB
              extracted_data = {};

              % Loop over each index in the list and extract corresponding data
              for idx = indices
                  % Extract the data for the given index (3D coordinates)
                  xyz_data = squeeze(data(idx, 1:3, :))';

                  % Format the XYZ data as a list in one column
                  formatted_column = arrayfun(@(row) sprintf('[%.4f, %.4f, %.4f]', xyz_data(row, 1), xyz_data(row, 2), xyz_data(row, 3)), ...
                                              1:size(xyz_data, 1), 'UniformOutput', false)';

                  % Add formatted column to extracted data
                  extracted_data = [extracted_data, formatted_column];
              end

              % Prepare column headers using the input file name and indices
              headers = {};
              for idx = indices
                  headers{end+1} = labels{idx};  % Use the label for the header
              end

              % Create a CSV file and write headers and data
              csv_file = fullfile(output_dir, ['SUB', num2str(j), '.csv']);
              fid = fopen(csv_file, 'w');
              fprintf(fid, '%s;', headers{1:end-1});
              fprintf(fid, '%s\n', headers{end});
              fclose(fid);

              % Append the formatted data
              fid = fopen(csv_file, 'a');
              [rows, cols] = size(extracted_data);

              for row = 1:rows
                  for col = 1:cols
                      fprintf(fid, '%s', extracted_data{row, col});
                      if col ~= cols
                          fprintf(fid, ';');
                      end
                  end
                  fprintf(fid, '\n');
              end
              fclose(fid);

              printf('Data from %s saved to %s\n', mat_file, csv_file);
              field_found = true;
              break;
            end
          end
          if ~field_found
              printf('Field aux_static_stoop_40 not found in %s\n', mat_file);
          end
    else
        printf('File %s not found\n', mat_file);
    end
end


