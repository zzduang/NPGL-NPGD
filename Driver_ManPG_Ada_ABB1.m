
function [XX,xopt, iter, comtime, fopt, nD, sparsity, avar, fs] = driver(x0, A, mu, L, tol, maxiter, Ftol,option)
    % parameters for line search
    %采用自适应非单调线搜索
    delta = 0.0001;
    gamma = 0.5;
    [m, n] = size(A);
    p = size(x0.main, 2);
    
    % functions for the optimization problem
    fhandle = @(x)f(x, A, mu);
    gfhandle = @(x)gf(x, A, mu);
    fprox = @prox;
    fcalJ = @calJ;
    
    % functions for the manifold
    fcalA = @calA;
    fcalAstar = @calAstar;
    
    xinitial = x0;
    tic
    [XX,xopt, fs, Ds] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, xinitial, L, tol, delta, gamma, maxiter, mu, Ftol,option);
    comtime = toc;
    xopt.main(abs(xopt.main) < 1e-5) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-5)) / (n * p);
    xopt = xopt.main;
    iter = length(fs);
    fopt = fs(end);
    nD = Ds;
%     fprintf('sparsity:%1.3f\n',sum(sum(abs(xopt.main) < 1e-5)) / (n * p));

    % adjusted variance
    [Q, R] = qr(A * xopt, 0);
    avar = trace(R * R);
end

function [XX,xopt, fs, Ds] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, x0, L, tol, delta, eta, maxiter, mu, Ftol,option)
    err = inf;
    x1 = x0;
    x2 = x1; 
    fs = []; 
    Ds = [];
    [f1, x1] = fhandle(x1);
    gf1 = gfhandle(x1);
    t = 1 / L;
    t0 = t;
    iter = 1;
    fs(iter) = f1;
    [n, p] = size(x0.main);
    Dinitial = zeros(p, p);
    totalbt = 0;
    %自适应非单调参数
    l = 0; LL=3;
    Fc = f1; 
    Fbest =f1; Fr = inf;
    
    num_inexact = 0;
    linesearch_flag = 0;
    t_min=1e-4;
    %M=option.M;
    alpha =1;  %线搜索求取步长
    
    DD=zeros(n,p);
   % Out.f = fs(iter);
    %Q = 1;Cval=fs(iter);
    while(err > tol && f1 > Ftol + 1e-7 && iter < maxiter)
        %子问题求解精度
        if alpha <t_min || num_inexact > 10 %或者子问题不精确求解次数大于10，调整子问题求解精度
            innertol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
        else
            innertol = max(1e-13, min(1e-11,1e-3*tol*t^2));
        end
        [D, Dinitial, inneriter] = finddir(x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dinitial, innertol);
%         alpha = 1;
%         x2 = R(x1, alpha * D);
%         [f2, x2] = fhandle(x2);
        
        %线搜索
%         while(f2 > f1 - delta * alpha * norm(D, 'fro')^2 && btiter < 3)
%             alpha = alpha * gamma;
%             x2 = R(x1, alpha * D);
%             [f2, x2] = fhandle(x2);
%             btiter = btiter + 1;
%             totalbt = totalbt + 1;
%         end
         % Nonmonotone line search algorithm
         
         normDsquared=norm(D, 'fro')^2;
         deriv=delta*normDsquared;
         %nls = 1;
         while 1
             x2 = R(x1, alpha * D); %迭代更新,%进行收缩
             [f2, x2] = fhandle(x2);
             %非单调线搜索
             if f2  <=Fr - 0.5/t* alpha * deriv  || nls >= 10 %
                 %btiter = btiter + 1;
                 totalbt = totalbt + 1;
                  break;
             else
                 alpha = eta*alpha;
                 nls = nls + 1;
             end
             
             %linesearch_flag = 1;
             %nls = nls + 1;%非单调线搜索
             if alpha<t_min
                 linesearch_flag = 1;
                 innertol = max(innertol * 1e-2, 1e-20);
                 continue;
             end
         end
         nls =0;
         %  update  f_r, f_c and f_best
         if f2 < Fbest
             Fbest = f2; Fc = f2; l = 0;
         else
             Fc = max(Fc,f2); l = l + 1;
             if l == LL
                 Fr = Fc;  Fc = f2; l = 0;
             end
         end
         
         
         % Compute the Alternate ODH step-size:
         S = x1.main-x2.main;             SS = sum(sum(S.*S));
         %XDiff = sqrt(SS/n);     FDiff = abs(F_trial-F(iter-1))/(abs(F(iter-1))+1);
         Y = D - DD;     SY = abs(sum(sum(S.*Y)));
        % if mod(iter,2)==0
        %     alpha = SS/abs(SY);
         %else
             YY = sum(sum(Y.*Y));
             alpha  = abs(SY)/YY;
         %end
         
         alpha = max(min(alpha, 1e10), 1e-10);
         %Raydan or HongChao Zhang
%          if strcmp(option.method, 'Raydan')
%              Cval = max( Out.f( iter+1- min(iter, M): iter+1) );
%          elseif strcmp(option.method, 'Hongchao')
%              % by HongChao Zhang
%              Qp = Q;
%              Q = gamma*Qp + 1;
%              Cval = (gamma*Qp*Cval +  f2)/Q;
%          end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        
        %赋值
        XX(:,:,iter)=x1;
        gf2 = gfhandle(x2);
        DD=D;
        fs(iter + 1) = f2;
        x1 = x2;
        f1 = f2; gf1 = gf2;
        iter = iter + 1;
        err =normDsquared/(t^(2)); %(norm(D, 'fro') / t)^2;
        Ds(iter) =err; %norm(D, 'fro');
        if(mod(iter, 100) == 0)
            fprintf('iter:%d, f:%e, err:%e, ngf:%e, inneriter:%d\n', iter, f1, err, norm(gf1, 'fro'), inneriter);
        end
        
        
        if linesearch_flag == 0
            t = t * 1.01;
        else
            t = max(t0,t / 1.01);
        end
    end
    fprintf('iter:%d, f:%e, err:%e, ngf:%e, totalbt:%d\n', iter, f1, err, norm(gf1, 'fro'), totalbt);
    xopt = x2;
end
function output = R(x, eta)
    [Q,R] = qr(x.main + eta,0);[U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function [output, x] = f(x, A, mu)
    x.Ax = A * x.main;
    tmp = norm(x.Ax, 'fro');
    output = - tmp * tmp + mu * sum(abs(x.main(:)));
end

function output = gf(x, A, mu)
    gfx = -2 * (A' * x.Ax);
    tmp = gfx' * x.main;
    output = gfx - x.main * ((tmp + tmp') / 2);
end

% compute E(Lambda)
function ELambda = E(Lambda, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
    if(length(BLambda) == 0)
        BLambda = x - t * (gfx - fcalAstar(Lambda, x));
    end
    DLambda = fprox(BLambda, t, mmu) - x;
    ELambda = fcalA(DLambda, x);
end

% compute calG(Lambda)[d]
function GLambdad = GLd(Lambda, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
        GLambdad = t * fcalA(fcalJ(BLambda, fcalAstar(d, x), t, mmu), x);
end

% Use semi-Newton to solve the subproblem and find the search direction
function [output, Lambda, inneriter] = finddir(xx, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu, x0, innertol)
    x = xx.main;
    lambda = 0.2;
    nu = 0.99;
    tau = 0.1;
    eta1 = 0.2; eta2 = 0.75;
    gamma1 = 3; gamma2 = 5;
    alpha = 0.1;
    beta = 1 / alpha / 100;
    [n, p] = size(x);
    
    z = x0;
    BLambda = x - t * (gfx - fcalAstar(z, x));
    Fz = E(z, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
    
    nFz = norm(Fz, 'fro');
    nnls = 5;
    xi = zeros(nnls, 1);% for non-monotonic linesearch
    xi(nnls) = nFz;
    maxiter = 1000;
    times = 0;
    Blocks = cell(p, 1);
    while(nFz * nFz > innertol && times < maxiter) % while not converge, find d and update z
        mu = lambda * max(min(nFz, 0.1), 1e-11);
        Axhandle = @(d)GLd(z, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu) + mu * d;
        [d, CGiter] = myCG(Axhandle, -Fz, tau, lambda * nFz, 30); % update d
        u = z + d;
        Fu = E(u, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu); 
        nFu = norm(Fu, 'fro');
        
        if(nFu < nu * max(xi))
            z = u;
            Fz = Fu;
            nFz = nFu;
            xi(mod(times, nnls) + 1) = nFz;
            status = 'success';
        else
            rho = - sum(Fu(:) .* d(:)) / norm(d, 'fro')^2;
            if(rho >= eta1)
                v = z - sum(sum(Fu .* (z - u))) / nFu^2 * Fu;
                Fv = E(v, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                nFv = norm(Fv, 'fro');
                if(nFv <= nFz)
                    z = v;
                    Fz = Fv;
                    nFz = nFv;
                    status = 'safegard success projection';
                else
                    z = z - beta * Fz;
                    Fz = E(z, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                    nFz = norm(Fz, 'fro');
                    status = 'safegard success fixed-point';
                end
            else
%                 fprintf('unsuccessful step\n');
                status = 'safegard unsuccess';
            end
            if(rho >= eta2)
                lambda = max(lambda / 4, 1e-5);
            elseif(rho >= eta1)
                lambda = (1 + gamma1) / 2 * lambda;
            else
                lambda = (gamma1 + gamma2) / 2 * lambda;
            end
        end
        BLambda = x - t * (gfx - fcalAstar(z, x));
%         fprintf(['iter:%d, nFz:%f, xi:%f, ' status '\n'], times, nFz, max(xi));
        times = times + 1;
    end
    Lambda = z;
    inneriter = times;
    output = fprox(BLambda, t, mmu) - x;
end

function output = prox(X, t, mu)
    output = min(0, X + t * mu) + max(0, X - t * mu);
end

function output = calA(Z, U) % U \in St(p, n)
    tmp = Z' * U;
    output = tmp + tmp';
end

function output = calAstar(Lambda, U) % U \in St(p, n)
    output = U * (Lambda + Lambda');
end

function output = calJ(y, eta, t, mu)
    output = (abs(y) > mu * t) .* eta;
end

function [output, k] = myCG(Axhandle, b, tau, lambdanFz, maxiter)
    x = zeros(size(b));
    r = b;
    p = r;
    k = 0;
    while(norm(r, 'fro') > tau * min(lambdanFz * norm(x, 'fro'), 1) && k < maxiter)
        Ap = Axhandle(p);
        alpha = r(:)' * r(:) / (p(:)' * Ap(:));
        x = x + alpha * p;
        rr0 = r(:)' * r(:);
        r = r - alpha * Ap;
        beta = r(:)' * r(:) / rr0;
        p = r + beta * p;
        k = k + 1;
    end
    output = x;
end
