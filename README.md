# solartoolbox
 A selection of tools used to analyse both real and simulated solar cells.

 ### ionmonger.py

 Contains classes for loading and analysing outputs from the IonMonger drift diffusion model. Takes the output of the following `MATLAB` script.

Modify your parameter input file to take `loop_item` as its argument and place `loop_item` as the variable you wish to iterate through.

  % Begin
  clear;
  tic;
  fprintf('Computation started at %s\n', datestr(now));
  reset_path();

  % Place list of variable to iterate through in var_list
  var_list = 1.2./logspace(-3,1);

  results = cell(length(var_list),2); % create a len(var_list) by 2 array

  for k = 1:length(var_list) % for i in len(var_list)
      loop_item = var_list(k); % MATLAB version of enumerate()
      disp(loop_item) % print(loop_item)
      results{k,1} = var_list(k); % place variable in column 1 of results
      params = parameters(loop_item); % make params struct using modified parameters file
      try % try to run simulation for iteration
          sol = numericalsolver(params); % solve for params given
          results{k,2} = struct(sol); % place solution for variable in column 2
          % plot_sections(sol,[2, 3]) % plot graph for visual check
          % axis([0 1.4 0 25]) % adjust axes
      catch me % if simulation failed, display error and move to next iteration
          disp('error')
      end
  end

  % Stop stopwatch and output nice message
  fprintf('Completed simulation at %s, taking %s\n', ...
      datestr(now), secs2hms(toc) )
