classdef aten__tanh4 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch
    %aten__tanh4 Auto-generated custom layer
    % Auto-generated by MATLAB on 14-Mar-2024 21:34:58
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        
        
        
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        
    end
    
    methods
        function obj = aten__tanh4(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 1;
            obj.NumOutputs = 1;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [tanh_3] = predict(obj,tanh_argument1_1)
            import model_scripted.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [tanh_argument1_1, tanh_argument1_1_format] = permuteToReversePyTorch(tanh_argument1_1, 'BCSS', 4);
            [tanh_argument1_1] = struct('value', tanh_argument1_1, 'rank', int64(4));
            
            % Placeholder function for aten::tanh.
            [tanh_3] = pyAtenTanh(tanh_argument1_1);
            
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(tanh_3.value) == 'U'))
                tanh_3 = permute(tanh_3.value, fliplr(1:max(2,tanh_3.rank)));
            end
            
        end
        
        
        
        function [tanh_3] = forward(obj,tanh_argument1_1)
            import model_scripted.ops.*;
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [tanh_argument1_1, tanh_argument1_1_format] = permuteToReversePyTorch(tanh_argument1_1, 'BCSS', 4);
            [tanh_argument1_1] = struct('value', tanh_argument1_1, 'rank', int64(4));
            
            % Placeholder function for aten::tanh.
            [tanh_3] = pyAtenTanh(tanh_argument1_1);
            
            %Permute U-labelled output to forward PyTorch dimension ordering
            if(any(dims(tanh_3.value) == 'U'))
                tanh_3 = permute(tanh_3.value, fliplr(1:max(2,tanh_3.rank)));
            end
            
        end
        
        
    end
end

