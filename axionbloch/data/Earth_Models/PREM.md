AI-extracted table data: 

**TABLE I**  
Coefficients of the polynomials describing the Preliminary Reference Earth Model (PREM).  
**Normalized radius**: \( x = r / a \) (where \(r\) = radius, \( a = 6371  \text{km} \)).  
**Applicable for**: Reference period of 1 second.

| Region             | Radius (km)     | Density (g cm\(^{-3}\))                  | \( V_P \) (km s\(^{-1}\))                 | \( V_S \) (km s\(^{-1}\))                 | \( Q_\mu \) | \( Q_\kappa \) |
| :----------------- | :-------------- | :--------------------------------------- | :---------------------------------------- | :---------------------------------------- | :---------- | :------------- |
| **Inner core**     | 0–1221.5        | \( 13.0885 - 8.8381x^2 \)                | \( 11.2622 - 6.3640x^2 \)                 | \( 3.6678 - 4.4475x^2 \)                  | 84.6        | 1327.7         |
| **Outer core**     | 1221.5–3480.0   | \( 12.5815 - 1.2638x - 3.6426x^2 - 5.5281x^3 \) | \( 11.0487 - 4.0362x + 4.8023x^2 - 13.5732x^3 \) | 0                                         | \( \infty \) | 57823          |
| **Lower mantle**   | 3480.0–3630.0   | \( 7.9565 - 6.4761x + 5.5283x^2 - 3.0807x^3 \) | \( 15.3891 - 5.3181x + 5.5242x^2 - 2.5514x^3 \) | \( 6.9254 - 1.4672x - 2.0834x^2 + 0.9783x^3 \) | 312         | 57823          |
|                    | 3630.0–5600.0   | \( 7.9565 - 6.4761x + 5.5283x^2 - 3.0807x^3 \) | \( 24.9520 - 40.4673x + 51.4832x^2 - 26.6419x^3 \) | \( 11.1671 - 13.7818x + 17.4575x^2 - 9.2777x^3 \) | 312         | 57823          |
|                    | 5600.0–5701.0   | \( 7.9565 - 6.4761x + 5.5283x^2 - 3.0807x^3 \) | \( 29.2766 - 23.6027x + 5.5242x^2 - 2.5514x^3 \) | \( 22.3459 - 17.2473x - 2.0834x^2 + 0.9783x^3 \) | 312         | 57823          |
| **Transition zone**| 5701.0–5771.0   | \( 5.3197 - 1.4836x \)                   | \( 19.0957 - 9.8672x \)                   | \( 9.9839 - 4.9324x \)                    | 143         | 57823          |
|                    | 5771.0–5971.0   | \( 11.2494 - 8.0298x \)                  | \( 39.7027 - 32.6166x \)                  | \( 22.3512 - 18.5856x \)                  | 143         | 57823          |
|                    | 5971.0–6151.0   | \( 7.1089 - 3.8045x \)                   | \( 20.3926 - 12.2569x \)                  | \( 8.9496 - 4.4597x \)                    | 143         | 57823          |
| **LVZ\***          | 6151.0–6291.0   | \( 2.6910 + 0.6924x \)                   | **Anisotropic**:<br>\( V_{PV} = 0.8317 + 7.2180x \)<br>\( V_{PH} = 3.5908 + 4.6172x \) | **Anisotropic**:<br>\( V_{SV} = 5.8582 - 1.4678x \)<br>\( V_{SH} = -1.0839 + 5.7176x \)<br>\( \eta = 3.3687 - 2.4778x \) | 80          | 57823          |
| **LID\***          | 6291.0–6346.6   | \( 2.6910 + 0.6924x \)                   | **Anisotropic**:<br>\( V_{PV} = 0.8317 + 7.2180x \)<br>\( V_{PH} = 3.5908 + 4.6172x \) | **Anisotropic**:<br>\( V_{SV} = 5.8582 - 1.4678x \)<br>\( V_{SH} = -1.0839 + 5.7176x \)<br>\( \eta = 3.3687 - 2.4778x \) | 600         | 57823          |
| **Crust**          | 6346.6–6356.0   | 2.900                                   | 6.800                                    | 3.900                                    | 600         | 57823          |
|                    | 6356.0–6368.0   | 2.600                                   | 5.800                                    | 3.200                                    | 600         | 57823          |
| **Ocean**          | 6368.0–6371.0   | 1.020                                   | 1.450                                    | 0                                        | \( \infty \) | 57823          |

---

### **Additional Notes**:
1. **LVZ/LID Anisotropy** (depths 24.4–220 km):  
   Effective isotropic approximations:  
   \( V_P = 4.1875 + 3.9382x \),  
   \( V_S = 2.1519 + 2.3481x \)
   
2. **Symbol Key**:  
   - \( V_{PV}/V_{PH} \): Vertical/Horizontal P-wave velocity  
   - \( V_{SV}/V_{SH} \): Vertical/Horizontal S-wave velocity  
   - \( \eta \): Anisotropy parameter  
   - \( Q_\mu \): Shear quality factor, \( Q_\kappa \): Bulk quality factor

> **Critical Features**:  
> - **Normalized radius \( x \)**: Explicitly defined as \( x = r / 6371 \) (non-dimensional)  
> - **Special Cases**: Outer core and Ocean have \( V_S = 0 \)  
> - **Constants**: Most regions use \( Q_\kappa = 57823 \)  
> - **Anisotropic Zones**: LVZ (Low Velocity Zone) and LID (Lithospheric Lid) require full parameter set  
> - **Expression Handling**: All polynomial terms use standard exponents (e.g., \( x^2 \), \( x^3 \)) as shown in original image  