function v_sat = vel_limiter(v, v_limit)
    v_temp    = sign(v) .* min(abs(v), v_limit);
    v_temp(1) = max(v_temp(1), 0);
    v_sat     = v_temp;
end
